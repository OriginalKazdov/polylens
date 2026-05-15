# Cookbook

Each recipe is a self-contained snippet. Run `pip install archscope` and pick what you need. Assumes `import archscope as ai` for brevity. All snippets target the public API surface of v0.2.7.

Recipes that touch Mamba run fine on CPU but are slow — installing `mamba-ssm` and running on CUDA enables the fast path. Anything labeled "transformer only" raises a clear error if you point it at an SSM.

## Recipes

1. [Extract the residual stream at every layer](#1-extract-the-residual-stream-at-every-layer)
2. [Extract Mamba's recurrent SSM state](#2-extract-mambas-recurrent-ssm-state)
3. [Train a linear probe on one layer](#3-train-a-linear-probe-on-one-layer)
4. [Apply a trained probe to externally-transformed activations](#4-apply-a-trained-probe-to-externally-transformed-activations)
5. [Run all behavioural circuits on any architecture](#5-run-all-behavioural-circuits-on-any-architecture)
6. [Logit lens with markdown output](#6-logit-lens-with-markdown-output)
7. [Train a tuned lens on calibration text](#7-train-a-tuned-lens-on-calibration-text)
8. [Activation patching: localize a behaviour to specific layers](#8-activation-patching-localize-a-behaviour-to-specific-layers)
9. [DIM decomposition: attention vs MLP contribution](#9-dim-decomposition-attention-vs-mlp-contribution)
10. [Cross-architecture probe transfer (Pythia to Mamba)](#10-cross-architecture-probe-transfer-pythia-to-mamba)
11. [Compare a base model against its fine-tuned version](#11-compare-a-base-model-against-its-fine-tuned-version)
12. [Train a sparse autoencoder on extracted activations](#12-train-a-sparse-autoencoder-on-extracted-activations)
13. [Run the InterpProfile benchmark from the CLI](#13-run-the-interpprofile-benchmark-from-the-cli)
14. [Register a custom backend for a non-supported architecture](#14-register-a-custom-backend-for-a-non-supported-architecture)

---

### 1. Extract the residual stream at every layer

`backend.layer_names()` returns the canonical handles your backend understands. Passing the full list to `backend.extract` walks all layers in a single forward pass and returns one `ActivationRecord` per layer with shape `(B, T, hidden_dim)`.

```python
import archscope as ai

model, tok, backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

inputs = tok(["The cat sat on the mat.", "Solve x squared minus 4."],
             return_tensors="pt", padding=True, truncation=True, max_length=24)

records = backend.extract(inputs, layers=backend.layer_names())
for rec in records:
    print(rec.layer_name, tuple(rec.activations.shape), rec.meta["kind"])
# layer_0.residual (2, 9, 768) residual
# layer_1.residual (2, 9, 768) residual
# ... one per block
```

See also: pass `layers=["layer_5.residual"]` if you only need a single layer — saves no compute (HF still runs the full forward) but reduces memory copying.

---

### 2. Extract Mamba's recurrent SSM state

The Mamba backend exposes two flavors of activations per block: `layer_N.residual` (the usual `(B, T, H)` stream) and `layer_N.ssm_state` — the final recurrent state after consuming the whole sequence. Shape is `(B, intermediate_size, ssm_state_size)`. This is the headline cross-architecture handle: the SSM state is Mamba's compressed "memory" and behaves nothing like a residual stream.

```python
import archscope as ai

model, tok, backend = ai.load_model("state-spaces/mamba-130m-hf", arch="mamba")

inputs = tok(["The cat sat on the mat.", "Music has the power to move us."],
             return_tensors="pt", padding=True, truncation=True, max_length=24)

rec = backend.extract(inputs, layers=["layer_12.ssm_state"])[0]
print(rec.layer_name, tuple(rec.activations.shape))
print("d_inner:", rec.meta["d_inner"], "d_state:", rec.meta["d_state"])
# layer_12.ssm_state (2, 1536, 16)
# d_inner: 1536 d_state: 16
```

Common gotcha: CPU is the HuggingFace reference path and is slow — install `mamba-ssm` and move the model to CUDA for production-speed extraction.

---

### 3. Train a linear probe on one layer

`fit_probe` has two calling conventions. The text-based one (shown here) handles tokenization and the kazdov-bool-mask detail for you. You get back a `ProbeFit` whose `.metrics` dict contains `train_auroc`, `val_auroc`, and `train_loss`.

```python
import archscope as ai

model, tok, backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

pf = ai.probes.fit_probe(
    model,
    tokenizer=tok,
    pos_texts=["I love this movie", "Wonderful day", "Amazing work"],
    neg_texts=["I hate this movie", "Terrible day", "Awful work"],
    layer_name="layer_5.residual",
    backend_hint="transformer",
)
print(pf.metrics)
# {'train_auroc': 1.0, 'val_auroc': 0.5, 'train_loss': 0.03...}
```

Common gotcha: with tiny `val_split` slices, `val_auroc` returns `0.5` (chance) when only one class is present in the split — bump dataset size before reading too much into validation numbers.

---

### 4. Apply a trained probe to externally-transformed activations

The DX shortcut from v0.2.7. For linear probes, `pf.direction` is the `(hidden_dim,)` weight vector and `pf.bias` is the scalar — together they let you score arbitrary tensors with `acts @ d + b` without going through the probe module. Useful when activations have been transformed (e.g., aligned across architectures) and you can't just call `pf.probe(acts)` anymore.

```python
import torch
import archscope as ai

pf = ai.probes.fit_probe(
    model,
    tokenizer=tok,
    pos_texts=["I love this", "Wonderful"],
    neg_texts=["I hate this", "Awful"],
    layer_name="layer_5.residual",
    backend_hint="transformer",
)

d, b = pf.direction, pf.bias        # shape (768,), shape ()

# Score a fresh activation tensor manually
new_acts = torch.randn(16, 768)
logits = new_acts @ d + b           # (16,)
probs = torch.sigmoid(logits)
```

Common gotcha: `.direction` raises `ValueError` on MLP probes — they have no single linear direction.

---

### 5. Run all behavioural circuits on any architecture

Every detector here is purely behavioral — it looks at model outputs given crafted inputs, never at attention weights — so the same code works on Pythia, Mamba, and any custom backend that returns logits. `run_all_circuits` is the one-call entry point.

```python
import archscope as ai

# Works the same way for "state-spaces/mamba-130m-hf" with arch="mamba"
model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

scores = ai.circuits.run_all_circuits(model, tokenizer=tok)
for name, cs in scores.items():
    print(f"{name:>30s}  score={cs.score:.4f}  relative={cs.relative:.2f}x")
# Actual Pythia-160m output:
#             induction_head  score=0.0097  relative=490.19x
#               copy_circuit  score=0.0000  relative=0.00x
#  early_token_concentration  score=6.1909  relative=0.57x
```

See also: the induction-head test does NOT need a tokenizer — it samples random vocab ids directly. Pass only `model` to get just that score.

---

### 6. Logit lens with markdown output

`logit_lens` applies the model's own final norm + unembedding to each layer's residual stream — the classic "what would the model commit to here?" projection. The returned `LensResult` has a `.to_markdown()` formatter that shows top-1 token, target probability, target rank, and entropy per layer.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

result = ai.lens.logit_lens(
    model, tok,
    prompt="The capital of France is",
    target_token=" Paris",
    backend_hint="transformer",
)
print(result.to_markdown())
# ### logit_lens on `The capital of France is`
# Target: ` Paris` (id=6342)
# | Layer | top-1 token | top-1 prob | target prob | rank | entropy |
# |-------|-------------|-----------:|------------:|-----:|--------:|
# |  0 | ` the`       | 0.012      | 0.000       | 38   | 10.21   |
# | ...
```

Works on Mamba too — pass `backend_hint="mamba"`. The lens reads the residual stream, which both backends expose under `layer_N.residual`.

---

### 7. Train a tuned lens on calibration text

The naive logit lens degrades sharply in mid-depth layers because the model's residual representation drifts away from the unembedding basis. The tuned lens (Belrose et al 2023) learns per-layer affine corrections from a small calibration set, then decodes more faithfully at every depth.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

calibration_texts = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees Celsius.",
    "Shakespeare wrote Hamlet.",
    "The mitochondria is the powerhouse of the cell.",
] * 4  # 16 short texts is enough for a small model

tl = ai.lens.TunedLens.fit(
    model, tok, calibration_texts,
    backend_hint="transformer",
    epochs=30, lr=1e-3,
)
print(f"trained, mean per-layer loss = {tl.last_loss:.4f}")

result = tl.predict(model, tok, "The capital of France is",
                    target_token=" Paris", backend_hint="transformer")
print(result.to_markdown())
```

Common gotcha: training uses each row's real last-token position via `attention_mask` — if you somehow pass calibration texts without a tokenizer that produces a mask, all rows are treated as full-length.

---

### 8. Activation patching: localize a behaviour to specific layers

Patch the residual stream from a "source" prompt into the corresponding layers of a "target" prompt, then measure how much of the source-vs-target behavioral gap the patch closes. `gap_restored = 1.0` means the patched layers fully explain the difference; values near `0.0` mean they don't.

```python
import torch
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

src = tok("The capital of France is", return_tensors="pt", padding=True)
tgt = tok("The largest city in Italy", return_tensors="pt", padding=True)
# Source/target must share input shape; pad to a common length yourself if not.

paris_id = tok(" Paris", add_special_tokens=False).input_ids[0]
rome_id  = tok(" Rome",  add_special_tokens=False).input_ids[0]

def logit_diff(out):
    last = out.logits[0, -1, :]
    return float(last[paris_id] - last[rome_id])

result = ai.attribute.activation_patch(
    model, src, tgt, layer_indices=[5, 6, 7],
    metric_fn=logit_diff, backend_hint="transformer",
)
print(f"layers {result.layer_range}: restored {result.gap_restored:.1%} of the gap")
# Actual Pythia-160m output (both prompts happen to be 5 tokens):
# layers (5, 7): restored 100.0% of the gap
```

Common gotcha: source and target `input_ids` must have the same shape — the patch hook installs the source residual directly. Pad or truncate to a common length.

---

### 9. DIM decomposition: attention vs MLP contribution

Splits a behavioral gap into per-component contributions by capturing each component's output during `prompt_a`, patching it into the forward on `prompt_b`, and reading the metric back. Returns one number per component, expressed as a fraction of the total gap.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

a = tok("The doctor said she", return_tensors="pt", padding=True)
b = tok("The doctor said he",  return_tensors="pt", padding=True)

def last_token_max_logit(out):
    return float(out.logits[0, -1, :].max())

dim = ai.attribute.dim_decompose(
    model, a, b, layer_indices=[4, 5, 6, 7],
    metric_fn=last_token_max_logit,
    components=("attention", "mlp"),
)
print(dim.components, "of total gap", round(dim.total, 3))
# Actual Pythia-160m output:
# {'attention': 1.019, 'mlp': 0.319} of total gap 0.443
```

Common gotcha: this is transformer-only. On Mamba it raises `ValueError` because there is no attention/MLP submodule — fall back to `activation_patch` on the residual stream instead.

---

### 10. Cross-architecture probe transfer (Pythia to Mamba)

Learn a ridge-regression alignment matrix `M` from paired activations, then transport a probe direction from the source space into the target space and score. The transferred probe is `w_target = M.T @ w_source`; bias is preserved.

```python
import torch
import archscope as ai
from archscope.transfer import learn_alignment

p_model, p_tok, p_back = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
m_model, m_tok, m_back = ai.load_model("state-spaces/mamba-130m-hf", arch="mamba")

align_texts = ["The cat sat on the mat.", "Music has power.",
               "Solve for x.", "Birds sing at dawn."] * 4

def pool_last_layer(backend, tok, texts, layer):
    inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=24)
    rec = backend.extract(inp, layers=[layer])[0]
    return rec.activations.mean(dim=1)        # (N, H)

src_acts = pool_last_layer(p_back, p_tok, align_texts, "layer_6.residual")
tgt_acts = pool_last_layer(m_back, m_tok, align_texts, "layer_12.residual")
M = learn_alignment(src_acts, tgt_acts, ridge=1e-3)   # (d_src, d_tgt)

# Suppose pf is a linear probe trained on Pythia (see recipe 3)
# w_mamba = M.T @ pf.direction; same bias
# Score Mamba activations directly: logits = mamba_acts @ w_mamba + pf.bias
```

See also: `examples/cross_arch_sentiment_transfer.py` for a full end-to-end Pythia ↔ Mamba sentiment-transfer study with stratified splits and 3-seed probes.

---

### 11. Compare a base model against its fine-tuned version

`diff.compare` walks the residual stream layer-by-layer on a small calibration set and reports L2 drift, relative drift, cosine similarity, and the top-shifted channels at each depth. By default it also re-runs the circuit detectors on both models and reports score deltas.

```python
import archscope as ai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m",
                                            dtype=torch.float32).eval()
ft   = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m",
                                            dtype=torch.float32).eval()
# (in practice, ft would be your fine-tuned checkpoint)

calibration = [
    "The cat sat on the mat.", "Solve for x in the equation.",
    "Music has the power to move us.", "Stars appeared in the dark sky.",
] * 4

result = ai.diff.compare(base, ft, tok, calibration, backend_hint="transformer")
print(result.to_markdown())
for d in result.top_changed_layers(3):
    print(f"layer {d.layer}: relative drift {d.relative_drift:.2%}, cos {d.cosine_similarity:.3f}")
```

Common gotcha: base and fine-tuned must share architecture and tokenizer. Use `run_circuits=False` to skip the circuit re-runs when iterating quickly.

---

### 12. Train a sparse autoencoder on extracted activations

`fit_sae` takes a flat `(N, input_dim)` tensor and a `SAEConfig`. Two flavors live behind the same API: `sae_type="dense"` is the standard SAE with L1 sparsity, `sae_type="rank1"` is the WriteSAE rank-1 factored variant (atoms are `v_i w_i^T` outer products) — built originally for recurrent cache writes. The returned model exposes `.last_metrics["recon"]`, `["l1"]`, and `["n_active"]`.

```python
import archscope as ai

model, tok, backend = ai.load_model("state-spaces/mamba-130m-hf", arch="mamba")

inputs = tok(["The cat sat on the mat.", "Solve x squared equals 16."],
             return_tensors="pt", padding=True, truncation=True, max_length=24)
rec = backend.extract(inputs, layers=["layer_12.ssm_state"])[0]
acts = rec.activations.reshape(rec.activations.shape[0], -1).detach()   # (B, d_inner*d_state)

cfg = ai.sae.SAEConfig(
    input_dim=acts.shape[-1], n_features=512,
    sae_type="dense",                     # or "rank1"
    sparsity=1e-4, learning_rate=3e-3,
)
sae = ai.sae.fit_sae(acts, cfg, epochs=60)
print(sae.last_metrics)
# Actual output with N=2 (toy):
# {'recon': 2.8e-07, 'l1': 5.2e-07, 'n_active': 0.020}
```

Real SAE numbers depend on (a) how many activation rows you feed (this 2-row toy will overfit), (b) `sparsity` weight, and (c) `n_features`. For meaningful results, accumulate `acts` across hundreds of token positions, raise `n_features` to a few thousand, and tune `sparsity`.

See also: rank-1 SAEs were proposed in WriteSAE for Mamba-style recurrent writes — worth comparing both on the same `ssm_state` activations if you're auditing SSM features.

---

### 13. Run the InterpProfile benchmark from the CLI

`archscope bench` is the one-command path to a standardized profile: probe AUROC at mid-depth for sentiment and math, three behavioral circuits, dense + rank-1 SAE reconstruction at a representative layer, and (for Mamba) SSM state variance ratio. Output format is inferred from the file extension.

```bash
# Markdown report to stdout
archscope bench EleutherAI/pythia-160m --arch transformer

# Markdown to a file
archscope bench EleutherAI/pythia-160m --arch transformer --out pythia.md

# JSON profile (machine-readable, dataclass-asdict shape)
archscope bench state-spaces/mamba-130m-hf --arch mamba --out mamba.json
```

Common gotcha: the SSM-state variance metric only populates when `--arch mamba`; transformer profiles leave that field as `NaN` by design. The CLI auto-picks a mid-depth `ssm_layer` for you.

---

### 14. Register a custom backend for a non-supported architecture

If your model is not in the autodetect table, you have two options. (a) For models that expose blocks via `model.blocks`, reuse `KazdovBackend` by passing `hint="kazdov"`. (b) For anything else, subclass `Backend` and override `layer_names`, `extract`, and `hidden_dim`. The `@Backend.register("name")` decorator wires it into the hint table.

```python
import torch
from archscope.backends import Backend, ActivationRecord

@Backend.register("myarch")
class MyArchBackend(Backend):
    def layer_names(self):
        return [f"layer_{i}.residual" for i in range(len(self.model.layers))]

    def extract(self, inputs, layers=None):
        layers = layers or self.layer_names()
        self._validate_layers(layers)
        captures = {}
        hooks = []
        for ln in layers:
            idx = int(ln.split("_")[1].split(".")[0])
            block = self.model.layers[idx]
            def make_hook(name):
                def hook(mod, inp, out):
                    tensor = out if isinstance(out, torch.Tensor) else out[0]
                    captures[name] = tensor.detach()
                return hook
            hooks.append(block.register_forward_hook(make_hook(ln)))
        try:
            with torch.no_grad():
                self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        finally:
            for h in hooks: h.remove()
        return [ActivationRecord(layer_name=ln, activations=captures[ln],
                                 meta={"kind": "residual", "arch": "myarch"})
                for ln in layers if ln in captures]

    def hidden_dim(self, layer_name):
        return self.model.config.hidden_size

# Then everywhere else in archscope: backend_hint="myarch"
# backend = Backend.for_model(my_model, hint="myarch")
```

See also: `src/archscope/kazdov_backend.py` is the working reference — about 60 lines, well-commented, supports any model with a `model.blocks` `ModuleList`.
