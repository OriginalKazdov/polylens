# archscope API reference — v0.2.7

Cross-architecture mechanistic interpretability for PyTorch / HuggingFace models. Every public symbol below is verified against `src/archscope/`.

Throughout this reference, snippets assume:

```python
import archscope as ai
```

---

## Quick reference table

| Symbol | One-line description |
|---|---|
| **Loading** | |
| `ai.load_model(name, arch=None, dtype=None, device="cpu")` | One-call HF model + tokenizer + backend |
| `ai.make_tokenize_fn(tokenizer, max_length=32, attention_mask_bool=False)` | Build a `texts -> dict` callable for backends |
| **Backends** | |
| `ai.backends.Backend` | Abstract base — `layer_names`, `extract`, `hidden_dim` |
| `ai.backends.Backend.for_model(model, hint=None)` | Autodetect or use hint to pick a backend |
| `ai.backends.Backend.register(name)` | Decorator to add a custom backend class |
| `ai.backends.ActivationRecord` | Dataclass: one layer's captured tensor + meta |
| `ai.backends.TransformerBackend` | Backend for HF decoder LMs (residual stream) |
| `ai.backends.MambaBackend` | Backend for Mamba / Mamba-2 (residual + `ssm_state`) |
| `ai.backends.RecurrentBackend` | Generic RNN backend (subclass to use) |
| `ai.kazdov_backend.KazdovBackend` | Backend for `model.blocks`-style custom architectures |
| **Probes** | |
| `ai.probes.ProbeConfig` | Config dataclass for one probe |
| `ai.probes.Probe` | `nn.Module` — linear or MLP head over hidden states |
| `ai.probes.ProbeFit` | Trains a `Probe`, exposes `.direction`, `.bias`, `.score` |
| `ai.probes.fit_probe(...)` | End-to-end: extract activations + fit probe |
| `ai.probes._auroc(logits, labels)` | AUROC utility, returns 0.5 on single-class splits |
| **SAEs** | |
| `ai.sae.SAEConfig` | Config for a sparse autoencoder |
| `ai.sae.DenseSAE` | Standard SAE: encoder + decoder + L1 |
| `ai.sae.Rank1FactoredSAE` | Rank-1 atom SAE (WriteSAE style) |
| `ai.sae.build_sae(config)` | Factory: returns the right SAE class |
| `ai.sae.fit_sae(activations, config, epochs=100, device="cpu")` | Train an SAE, attach `last_metrics` |
| **Neurons** | |
| `ai.neurons.NeuronEditConfig` | Config: `top_frac`, `layer_filter`, `mode` |
| `ai.neurons.NeuronEdit` | Result of contrastive search — `.apply_hook(model)` |
| `ai.neurons.find_neurons(model, inputs_harmful, inputs_benign, ...)` | Contrastive pair search |
| **Attribute** | |
| `ai.attribute.PatchResult` | Dataclass: outcome of a patching experiment |
| `ai.attribute.DIMResult` | Dataclass: per-component (attn/mlp) contributions |
| `ai.attribute.activation_patch(...)` | Patch source activations into target forward |
| `ai.attribute.dim_decompose(...)` | Difference-in-means per component (transformer-only) |
| **Circuits** | |
| `ai.circuits.CircuitScore` | Dataclass: `name, score, baseline, relative, raw` |
| `ai.circuits.induction_head_score(model, ...)` | Olsson induction test |
| `ai.circuits.copy_score(model, tokenizer, ...)` | Verbatim copy test |
| `ai.circuits.early_token_attention(model, tokenizer, ...)` | Attention-sink proxy via entropy |
| `ai.circuits.run_all_circuits(model, tokenizer=None, ...)` | Run all available tests |
| **Lens** | |
| `ai.lens.LayerPrediction` | Per-layer top-k tokens + entropy |
| `ai.lens.LensResult` | List of `LayerPrediction` + `to_markdown()` |
| `ai.lens.logit_lens(model, tokenizer, prompt, ...)` | Nostalgebraist logit lens |
| `ai.lens.TunedLens` | Learned per-layer affine translators (Belrose et al) |
| **Diff** | |
| `ai.diff.LayerDrift` | Per-layer L2 / cosine / top-shifted-neuron drift |
| `ai.diff.CircuitDelta` | Circuit-score change (base vs fine-tuned) |
| `ai.diff.ModelDiff` | Full diff + `.top_changed_layers(k)` + `.to_markdown()` |
| `ai.diff.compare(base, ft, tokenizer, calibration_texts, ...)` | Base vs fine-tuned comparison |
| **Transfer** | |
| `ai.transfer.TransferResult` | Dataclass: in-arch baselines + transferred AUROC |
| `ai.transfer.learn_alignment(src, tgt, ridge=1e-3)` | Ridge regression alignment matrix |
| `ai.transfer.transfer_probe(w, b, M)` | Move a linear probe across alignment |
| `ai.transfer.evaluate_transfer(...)` | Full source→target transfer pipeline |
| `ai.transfer.auroc_from_scores(scores, labels)` | AUROC helper |
| **Bench** | |
| `ai.bench.InterpProfile` | Dataclass: standardized interp scores |
| `ai.bench.benchmark(model_name, model, tokenizer, ...)` | Run InterpBench |
| `ai.bench.profile_to_markdown(profile)` | Human-readable report |
| **CLI** | |
| `archscope info` | Print methods + backends table |
| `archscope bench MODEL_NAME --arch ARCH [--out PATH]` | Run InterpBench on a HF model |

---

## archscope.load_model + make_tokenize_fn

### `ai.load_model(name, arch=None, dtype=None, device="cpu") -> (model, tokenizer, backend)`

Loads a HuggingFace model + tokenizer and returns them together with a matching `Backend`. Sets `tokenizer.pad_token = eos_token` when missing, defaults `dtype` to `torch.float32`, and calls `model.eval()`. If `arch` is `None`, autodetect from `config.model_type`; pass `arch="transformer"|"mamba"|"kazdov"|"recurrent"` explicitly for any model not in the autodetect table.

### `ai.make_tokenize_fn(tokenizer, max_length=32, attention_mask_bool=False) -> Callable[[list[str]], dict]`

Returns a function that tokenizes a list of strings with `return_tensors="pt"`, `padding=True`, `truncation=True`. If `attention_mask_bool=True`, the returned `attention_mask` is cast to `bool` (required for the `kazdov` backend; HF default is `int64`).

```python
import archscope as ai

model, tok, backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
tokenize = ai.make_tokenize_fn(tok, max_length=32)

inputs = tokenize(["The capital of France is", "Music is the food of"])
records = backend.extract(inputs, layers=["layer_5.residual"])
print(records[0].activations.shape)  # (2, T, hidden)
```

---

## archscope.backends

### `ActivationRecord` (dataclass)

| Field | Type | Description |
|---|---|---|
| `layer_name` | `str` | Identifier, e.g. `"layer_5.residual"` |
| `activations` | `torch.Tensor` / `jax.Array` | Captured tensor |
| `meta` | `dict` | Arch-specific info, e.g. `{"kind": "residual", "arch": "transformer"}` |

### `Backend` (abstract)

Subclasses must implement:

- `layer_names() -> list[str]` — virtual layer handles consumed by `extract`.
- `extract(inputs, layers=None) -> list[ActivationRecord]` — returns one record per requested layer; `inputs` is the dict from `make_tokenize_fn`.
- `hidden_dim(layer_name: str) -> int` — dimensionality at that layer.

Class methods:

- `Backend.for_model(model, hint=None) -> Backend` — autodetect or use hint. Raises `ValueError` when neither matches.
- `Backend.register(name)` — class decorator to register a custom backend under that name.

### Autodetect table

`Backend.for_model(model)` reads `model.config.model_type` and maps to a backend. The shipped table covers 19 model_types:

| model_type | backend |
|---|---|
| `llama` | transformer |
| `mistral` | transformer |
| `qwen2` | transformer |
| `qwen3` | transformer |
| `gpt2` | transformer |
| `gpt_neox` (Pythia) | transformer |
| `gpt_neo` | transformer |
| `gptj` | transformer |
| `falcon` | transformer |
| `mpt` | transformer |
| `bloom` | transformer |
| `opt` | transformer |
| `phi` | transformer |
| `phi3` | transformer |
| `gemma` | transformer |
| `gemma2` | transformer |
| `starcoder2` | transformer |
| `mamba` | mamba |
| `mamba2` | mamba |

Anything else needs an explicit `hint=...` or a custom `Backend.register("name")`.

### `TransformerBackend`

- `layer_names()` → `["layer_0.residual", ..., "layer_{n-1}.residual"]` where `n = config.num_hidden_layers`.
- `extract(inputs, layers=None)` → runs the model with `output_hidden_states=True` under `torch.no_grad()`. Each record's `activations` has shape `(B, T, hidden_size)`; meta is `{"kind": "residual", "arch": "transformer"}`.
- `hidden_dim(layer_name)` → `config.hidden_size`.

### `MambaBackend`

Two layer-name conventions per block:

- `layer_N.residual` → residual stream after block `N`, shape `(B, T, hidden_size)`. Meta: `{"kind": "residual", "arch": "mamba", "shape_meaning": "(B, T, hidden_size)"}`.
- `layer_N.ssm_state` → final SSM recurrent state after processing the sequence, shape `(B, intermediate_size, ssm_state_size)`. Meta: `{"kind": "ssm_state", "arch": "mamba", "shape_meaning": "(B, intermediate_size, ssm_state_size)", "d_inner": ..., "d_state": ...}`.

`extract` passes a `DynamicCache(config=...)` only when at least one requested layer is an `ssm_state`; with `use_cache=True` Mamba writes final SSM states into `cache.layers[idx].recurrent_states`, which `extract` detaches and returns.

`hidden_dim("layer_N.ssm_state")` returns `intermediate_size * state_size` (falling back to introspecting `model.backbone.layers[0].mixer`). `hidden_dim("layer_N.residual")` returns `config.hidden_size` (or `config.d_model`).

### `RecurrentBackend`

Generic. Default `layer_names()` returns `[f"layer_{i}.hidden" for i in range(model.n_layer or model.num_layers)]`, or `["layer_0.hidden"]` if neither exists. Default `extract` calls `model.get_hidden_states(inputs)` (a `dict[str, tensor]`) — if the model does not expose that method, subclass and override. `hidden_dim` tries `d_model`, `hidden_size`, `d_hidden`, `n_embd` on the model.

### `KazdovBackend` (in `archscope.kazdov_backend`, registered as `"kazdov"`)

Generic backend for any PyTorch model that exposes:

- `model.blocks` (a `nn.ModuleList` of residual blocks)
- `model.d_model` or `model.hidden_size`
- forward signature `model(input_ids, attention_mask=None, ...)`

`layer_names()` returns `[f"layer_{i}.residual" for i in range(len(model.blocks))]`. `extract` installs forward hooks on each requested block, runs a forward pass, then removes the hooks. Each captured tensor is detached. Meta: `{"kind": "residual", "arch": "kazdov-blocks"}`. `hidden_dim` reads `d_model` then `hidden_size`.

```python
import archscope as ai

model, tok, backend = ai.load_model("state-spaces/mamba-130m-hf", arch="mamba")
tokenize = ai.make_tokenize_fn(tok)
recs = backend.extract(tokenize(["Hello world"]),
                       layers=["layer_5.residual", "layer_5.ssm_state"])
for r in recs:
    print(r.layer_name, tuple(r.activations.shape), r.meta["kind"])
```

---

## archscope.probes

### `ProbeConfig` (dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `layer_name` | `str` | — | Layer to probe (e.g., `"layer_5.residual"`) |
| `probe_type` | `str` | `"linear"` | `"linear"` or `"mlp"` |
| `hidden_dim` | `int` | `64` | MLP hidden width (ignored for linear) |
| `target` | `str` | `"performativity"` | Label for the probed property |

### `Probe(nn.Module)`

Linear (`nn.Linear(input_dim, 1)`) or MLP (`Linear → GELU → Linear`) head. `forward(x)` accepts `(N, hidden_dim)` or `(N, seq, hidden_dim)` and returns logits with the final dim collapsed to a scalar.

### `ProbeFit(config, input_dim, device="cpu")`

Methods:

- `train(activations, labels, epochs=50, lr=1e-3, batch_size=64, val_split=0.2) -> dict` — supervised fit with `AdamW` + `BCEWithLogitsLoss`. If `activations.dim() == 3`, pools the sequence dim by mean. Returns `{"train_auroc": float, "val_auroc": float, "train_loss": float}`.
- `score(activations) -> torch.Tensor` — applies `sigmoid(probe(x))`, per-example or per-token depending on input shape.

Properties (linear probes only — both raise `ValueError` for MLP):

- `direction` → `torch.Tensor` of shape `(hidden_dim,)`. The probe's projection axis in activation space. Useful for transferring probes after `archscope.transfer.learn_alignment` or projecting interventions.
- `bias` → scalar `torch.Tensor`. Together with `direction`, scoring becomes `logits = acts @ direction + bias`.

### `fit_probe(model, inputs_pos=None, inputs_neg=None, layer_name="", backend_hint=None, config=None, device="cpu", *, tokenizer=None, pos_texts=None, neg_texts=None, max_length=32) -> ProbeFit`

End-to-end probe-fitting helper. Two calling conventions:

1. **Pre-tokenized**: pass `inputs_pos` and `inputs_neg` dicts (with `input_ids`, optional `attention_mask`).
2. **Texts + tokenizer**: pass `tokenizer=`, `pos_texts=[...]`, `neg_texts=[...]`. Archscope tokenizes via `make_tokenize_fn`; if `backend_hint == "kazdov"`, the attention_mask is auto-cast to bool.

Internally: extracts pos/neg activations at `layer_name`, mean-pools the seq dim if 3-D, labels pos=1 / neg=0, trains via `ProbeFit.train`. The returned `ProbeFit` has a `metrics` attribute attached (`train_auroc`, `val_auroc`, `train_loss`).

```python
import archscope as ai

model, tok, backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
pf = ai.probes.fit_probe(
    model,
    layer_name="layer_5.residual",
    backend_hint="transformer",
    tokenizer=tok,
    pos_texts=["I love this movie", "What a wonderful day"],
    neg_texts=["I hate this place", "What a terrible movie"],
)
print(pf.metrics)
print("direction shape:", pf.direction.shape)
```

### `_auroc(logits, labels) -> float`

Internal helper. Converts `logits` via `sigmoid` and calls `sklearn.metrics.roc_auc_score`. Edge case: if only one class is present in `labels`, emits a warning and returns `0.5` (chance) rather than NaN, so very small validation splits don't crash downstream code.

---

## archscope.sae

### `SAEConfig` (dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `input_dim` | `int` | — | Activation dimension |
| `n_features` | `int` | — | Dictionary size |
| `sparsity` | `float` | `1e-3` | L1 coefficient |
| `sae_type` | `str` | `"dense"` | `"dense"` or `"rank1"` |
| `learning_rate` | `float` | `1e-3` | AdamW lr |
| `batch_size` | `int` | `64` | Training batch size |

### `DenseSAE(config)`

Standard SAE. `encoder = Linear(input_dim, n_features)`, `decoder = Linear(n_features, input_dim, bias=False)`. Decoder weight is initialized as `encoder.weight.T`. Forward returns `(x_hat, z)` where `z = relu(encoder(x))`. `loss(x)` returns `(loss, {"recon": float, "l1": float, "n_active": float})`.

### `Rank1FactoredSAE(config)`

WriteSAE-style rank-1 atoms. Two parameters: `v` and `w`, each `(n_features, input_dim)`, initialized as `randn * 0.02`. `fire(x) = relu(x @ w.T)`. `reconstruct(x) = fire(x) @ v`. Asserts `config.sae_type == "rank1"`. Designed for recurrent cache writes where atoms substitute for native write contributions at matched Frobenius norm.

### `build_sae(config) -> nn.Module`

Factory: returns `DenseSAE` for `"dense"`, `Rank1FactoredSAE` for `"rank1"`. Raises `ValueError` otherwise.

### `fit_sae(activations, config, epochs=100, device="cpu") -> nn.Module`

Trains the SAE on flattened activations `(N, input_dim)` using `AdamW(lr=config.learning_rate)`, no validation split. After training, attaches `sae.last_metrics = {"recon": float, "l1": float, "n_active": float}` from the final mini-batch.

### Choosing `sae_type`

- **`"dense"`** — default. Works on transformer residual streams and any pooled activation. Pick this unless you specifically want atom geometry tied to recurrent writes.
- **`"rank1"`** — WriteSAE atoms `v_i w_i^T`. Built for recurrent CACHE writes; the firing strength is `<w_i, x>` and the output contribution is `fires_i * v_i`. Use this on Mamba `ssm_state` or RNN hidden states where the "write" interpretation is meaningful.

```python
import torch, archscope as ai

acts = torch.randn(2000, 768)
cfg = ai.sae.SAEConfig(input_dim=768, n_features=512, sae_type="dense",
                       sparsity=1e-4, learning_rate=3e-3)
sae = ai.sae.fit_sae(acts, cfg, epochs=60)
print(sae.last_metrics)  # {"recon": ..., "l1": ..., "n_active": ...}
```

---

## archscope.neurons

### `NeuronEditConfig` (dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `top_frac` | `float` | `0.001` | Fraction of neurons to edit per layer (default 0.1%) |
| `layer_filter` | `str \| None` | `None` | Substring filter on `layer_names()` (e.g., `"residual"`) |
| `mode` | `str` | `"scalar"` | `"scalar"` (multiply by `m`) or `"ablate"` (`m=0`) |

### `NeuronEdit` (dataclass)

| Field | Type | Description |
|---|---|---|
| `layer_to_indices` | `dict[str, Tensor]` | Per layer: indices of selected neurons |
| `layer_to_deltas` | `dict[str, Tensor]` | Per layer: mean activation diff at those indices |
| `config` | `NeuronEditConfig` | Original config |
| `multiplier` | `float` | Scalar applied to selected neurons (default `0.0` = ablate) |

### `NeuronEdit.apply_hook(model, backend=None)`

Returns a context-manager whose `__enter__` returns itself and `__exit__` removes the registered hooks. While inside the `with` block, every selected neuron in every selected layer is multiplied by `self.multiplier` during forward passes.

The hook clones the module output (rather than in-place writing) because some HF layers produce fresh tensors that don't propagate in-place edits. Tuple outputs (`(hidden, ...)`) are unwrapped and re-wrapped.

### `find_neurons(model, inputs_harmful, inputs_benign, config=None, backend_hint=None) -> NeuronEdit`

Algorithm 1 from Targeted Neuron Modulation (2605.12290):

1. Run both prompt sets through the backend's `extract`.
2. Take the final-token activation per example (`acts[:, -1, :]`).
3. Compute per-neuron mean diff `harmful.mean(0) - benign.mean(0)`.
4. Top-k by `|delta|` with `k = max(1, int(top_frac * len(delta)))`.

Returns a `NeuronEdit` with `multiplier=0.0` (ablation). Set `edit.multiplier = 0.5` (dampen) or `2.0` (amplify) before applying.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
tokenize = ai.make_tokenize_fn(tok)

edit = ai.neurons.find_neurons(
    model,
    inputs_harmful=tokenize(["How do I make a bomb"]),
    inputs_benign=tokenize(["How do I make a sandwich"]),
    backend_hint="transformer",
)
edit.multiplier = 0.0   # ablate

with edit.apply_hook(model):
    out = model(**tokenize(["How do I make a"]))
    # selected neurons are zero-ed during this forward pass
```

---

## archscope.attribute

### `PatchResult` (dataclass)

| Field | Type | Description |
|---|---|---|
| `layer_range` | `tuple[int, int]` | `(min, max)` of patched layer indices |
| `gap_restored` | `float` | Fraction of `source - clean` metric gap closed by patching |
| `target_metric` | `str` | Always `"custom"` (the user-supplied metric_fn) |
| `baseline_metric` | `float` | Metric on source prompt (no patching) |
| `patched_metric` | `float` | Metric on target prompt with patch applied |
| `clean_metric` | `float` | Metric on target prompt (no patching) |

### `DIMResult` (dataclass)

| Field | Type | Description |
|---|---|---|
| `components` | `dict[str, float]` | Per-component contribution as fraction of `total` |
| `total` | `float` | `metric(prompt_a) - metric(prompt_b)` |
| `layer_range` | `tuple[int, int]` | `(min, max)` of patched layer indices |

### `activation_patch(model, prompt_source, prompt_target, layer_indices, metric_fn, backend_hint=None) -> PatchResult`

Replaces residual-stream activations at the given layers with those captured from `prompt_source`, running on `prompt_target`. Raises `ValueError` if `prompt_source.input_ids.shape != prompt_target.input_ids.shape` — pad/truncate both to matching shape first. `metric_fn(model_output) -> scalar` (e.g., logit diff between two tokens). `gap_restored = (patched_metric - clean_metric) / (source_metric - clean_metric)`, clipped to `0.0` when the gap is below `1e-9`.

### `dim_decompose(model, prompt_a, prompt_b, layer_indices, metric_fn, components=("attention", "mlp"), backend_hint=None) -> DIMResult`

For each component (default `"attention"` then `"mlp"`): hook the submodule on `prompt_a` to capture its output, then patch that exact output into `prompt_b`'s forward pass, measure the metric, and store `(patched - metric_b) / (total_gap + 1e-9)` in `components[comp]`.

**Caveat — transformer-only:** raises `ValueError` upfront when none of the requested components are resolvable on this model. Mamba, pure SSMs, and custom recurrent blocks do not expose `self_attn`/`attn`/`attention` or `mlp`/`feed_forward`/`ffn` submodules; for those architectures use `activation_patch` on the residual stream instead. The error message says exactly that.

`backend_hint` is accepted only for API symmetry and is unused inside `dim_decompose` (the function resolves modules directly via `_utils.resolve_subcomponent_module`).

```python
import torch, archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

def metric(out):
    return out.logits[0, -1, :].max().item()  # peak last-token logit

src = tok("The capital of France is", return_tensors="pt", padding="max_length", max_length=8)
tgt = tok("The capital of Spain is", return_tensors="pt", padding="max_length", max_length=8)

res = ai.attribute.activation_patch(model, src, tgt, list(range(4, 8)), metric)
print(res.gap_restored, res.layer_range)
```

---

## archscope.circuits

All three detectors are **behavioural** — they only need a forward pass, no internals — so they work uniformly on transformer, Mamba, hybrid and custom architectures.

### `CircuitScore` (dataclass)

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Detector name |
| `score` | `float` | Primary metric (higher = circuit more present) |
| `baseline` | `float` | Random-model baseline |
| `relative` | `float` | `score / baseline` (>1 = circuit present) |
| `raw` | `dict` | Detector-specific detail numbers |

### `induction_head_score(model, n_pairs=20, seq_len=6, n_trials=50, vocab_size=None, device="cpu", seed=0) -> CircuitScore`

Olsson-style test. Constructs `[A1 B1] [A2 B2] ... [A_k B_k] [A_1]` and measures `P(next token = B_1)`. Infers `vocab_size` from `model.config.vocab_size` → `model.vocab_size` → `50257`. Uses a token window `[lo, hi)` with `lo = min(100, vocab_size//4)` and `hi = min(vocab_size, 40000)`; raises `ValueError` if the window is too small for `2 * n_pairs` distinct ids.

Returned `raw`: `accuracy_top1`, `avg_rank_target`, `avg_prob_target`, `n_trials`, `seq_len_pairs`.

### `copy_score(model, tokenizer, n_trials=30, n_words=5, device="cpu", seed=0) -> CircuitScore`

Tests verbatim copy: feeds `"list: A B C D E. list: "` and autoregressively predicts `n_words` tokens, scoring fraction equal to the input list. Handles BPE (`" word"`) and SentencePiece (`"▁word"`) by trying `" "+word` first, falling back to bare `word`. Predictions are chained (not teacher-forced) so this measures cumulative copy ability, not single-step accuracy.

Returned `raw`: `n_trials`, `n_words`, `correct`, `total`.

### `early_token_attention(model, tokenizer, texts=None, device="cpu") -> CircuitScore`

Coarse behavioural proxy for the "attention sink" phenomenon. For each text, computes Shannon entropy of the next-token distribution (in nats). `score` = mean entropy across texts; `baseline` = `log(vocab_size)` (max entropy); `relative = score / baseline` where `0` is full concentration and `1` is uniform. If `texts` is `None`, uses 8 built-in defaults. This is a behavioural proxy — real attention-sink analysis needs architecture-specific attention weights, but the proxy works on any architecture (including SSMs without attention).

Returned `raw`: `per_text_entropy`, `max_entropy`.

### `run_all_circuits(model, tokenizer=None, device="cpu") -> dict[str, CircuitScore]`

Runs `induction_head` unconditionally. If `tokenizer` is provided, also runs `copy_circuit` and `early_token_concentration`. Returns a dict keyed by circuit name.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
results = ai.circuits.run_all_circuits(model, tokenizer=tok)
for name, s in results.items():
    print(name, f"score={s.score:.3f}", f"baseline={s.baseline:.3e}", f"x={s.relative:.1f}")
```

---

## archscope.lens

### `LayerPrediction` (dataclass)

| Field | Type | Description |
|---|---|---|
| `layer` | `int` | Layer index |
| `layer_name` | `str` | e.g., `"layer_5.residual"` |
| `top_tokens` | `list[tuple[int, str, float]]` | `(token_id, token_str, prob)` top-k |
| `target_prob` | `float \| None` | Prob of target token at this layer (if `target_token` set) |
| `target_rank` | `int \| None` | Rank of target token (0 = top-1) |
| `entropy` | `float` | Shannon entropy in nats |

### `LensResult` (dataclass)

| Field | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt |
| `target_token` | `str \| None` | Target token string (if provided) |
| `target_token_id` | `int \| None` | First token id of `target_token` |
| `layers` | `list[LayerPrediction]` | One entry per layer probed |
| `method` | `str` | `"logit_lens"` or `"tuned_lens"` |

Method: `to_markdown() -> str` — renders a per-layer markdown table with top-1 token, top-1 prob, target prob, target rank, entropy.

### `logit_lens(model, tokenizer, prompt, target_token=None, layers=None, backend_hint=None, top_k=5, device="cpu") -> LensResult`

Nostalgebraist logit lens. For each layer, takes the last-token residual, applies the model's own final norm (located via `_utils.resolve_final_norm` — Llama `model.norm`, Pythia `gpt_neox.final_layer_norm`, GPT-2/Falcon `transformer.ln_f`, Mamba `backbone.norm_f`, top-level `ln_f` for kazdov), then unembeds (located via `_utils.resolve_unembedding` — `lm_head`, `embed_out`, or `output_layer`). Returns one `LayerPrediction` per residual layer.

Caveat — on Mamba, logit lens fidelity tends to degrade with depth because the residual stream is only one of two channels (the SSM state carries the rest of the model's working memory); typical reading: shallow layers are interpretable, deep layers less so. The function itself doesn't care; the interpretation is on the caller.

### `TunedLens(nn.Module)` — `TunedLens.fit(...)`, `TunedLens.predict(...)`

Belrose et al 2023. One `Linear(hidden_dim, hidden_dim, bias=True)` per layer, initialized to identity (so an untrained tuned lens equals a logit lens).

`TunedLens.fit(model, tokenizer, calibration_texts, backend_hint=None, epochs=30, lr=1e-3, max_len=64, device="cpu") -> TunedLens`:

- Tokenizes all calibration texts in one batch (`padding=True`, `truncation=True`, `max_length=max_len`); sets `pad_token = eos_token` if missing.
- Extracts residual activations once.
- Computes "real last position" per row from the attention mask (so padding is not used as the target position).
- Target = `softmax(unembed(norm(last_layer_residual_at_real_last_pos)))`.
- For each layer, learns `translator_i` via `kl_div(log_softmax(unembed(norm(translator_i(x)))), target_log_probs, log_target=True, reduction="batchmean")` summed across layers.
- Stores `tl.last_loss = total_loss / n_layers` at the end.

`TunedLens.predict(model, tokenizer, prompt, target_token=None, backend_hint=None, layers=None, top_k=5, device="cpu") -> LensResult`:

- Same flow as `logit_lens`, but each layer's residual is run through the learned `translator_i` before norm + unembed.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")

# Logit lens — no training
res = ai.lens.logit_lens(model, tok, "The capital of France is",
                         target_token=" Paris", backend_hint="transformer")
print(res.to_markdown())

# Tuned lens — fit on calibration corpus first
calib = ["The cat sat on the", "Music is the food of"] * 8
tl = ai.lens.TunedLens.fit(model, tok, calib, backend_hint="transformer", epochs=30)
res2 = tl.predict(model, tok, "The capital of France is",
                  target_token=" Paris", backend_hint="transformer")
```

---

## archscope.diff

### `LayerDrift` (dataclass)

| Field | Type | Description |
|---|---|---|
| `layer` | `int` | Layer index |
| `layer_name` | `str` | e.g., `"layer_5.residual"` |
| `mean_l2_delta` | `float` | Mean L2 norm of per-token `ft - base` |
| `relative_drift` | `float` | `mean_l2_delta / mean ||base||` |
| `cosine_similarity` | `float` | Mean per-token `cos(base, ft)` |
| `top_shifted_neurons` | `list[tuple[int, float]]` | Top channels by mean absolute delta |

### `CircuitDelta` (dataclass)

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Circuit name |
| `base_score` | `float` | Score on base model |
| `fine_tuned_score` | `float` | Score on fine-tuned model |
| `delta` | `float` | `ft - base` |
| `relative_change` | `float` | `delta / (|base| + 1e-9)` |

### `ModelDiff` (dataclass)

| Field | Type | Description |
|---|---|---|
| `arch_family` | `str` | Backend hint used (or `"auto"`) |
| `n_layers` | `int` | Number of residual layers compared |
| `n_calibration_texts` | `int` | Size of calibration corpus |
| `layer_drift` | `list[LayerDrift]` | Per-layer drift |
| `circuit_deltas` | `list[CircuitDelta]` | Per-circuit deltas |
| `notes` | `list[str]` | Free-form warnings/errors |

Methods:

- `top_changed_layers(k=3) -> list[LayerDrift]` — k layers with highest `relative_drift`, sorted descending.
- `to_markdown() -> str` — formatted multi-section report (per-layer table + top-3 layers + circuit deltas + notes).

### `compare(base_model, fine_tuned_model, tokenizer, calibration_texts, backend_hint=None, run_circuits=True, max_length=32, device="cpu") -> ModelDiff`

Both models must share architecture and tokenizer (validated by comparing `layer_names()` filtered to `.residual`). Sets `pad_token = eos_token` when missing. Tokenizes `calibration_texts` once and reuses for both. If `backend_hint == "kazdov"`, casts `attention_mask` to bool. Per-layer drift is computed by `_activation_drift` (mean L2 delta + cosine similarity + top-10 channels by mean absolute delta). When `run_circuits=True`, calls `circuits.run_all_circuits` on each model and stores deltas; any exception is captured into `notes` rather than propagated.

```python
import archscope as ai

base = ai.load_model("EleutherAI/pythia-160m", arch="transformer")[0]
ft, tok, _ = ai.load_model("./my-finetuned-pythia-160m", arch="transformer")

calib = ["The cat sat", "Music is", "Mountains"] * 8
diff = ai.diff.compare(base, ft, tok, calib, backend_hint="transformer")
print(diff.to_markdown())
print("most-changed:", [d.layer for d in diff.top_changed_layers(3)])
```

---

## archscope.transfer

### `TransferResult` (dataclass)

| Field | Type | Description |
|---|---|---|
| `source_arch` | `str` | Source architecture name passed in |
| `target_arch` | `str` | Target architecture name passed in |
| `source_layer` | `str` | Layer name on source |
| `target_layer` | `str` | Layer name on target |
| `n_align_pairs` | `int` | Number of paired-text examples used to fit `M` |
| `baseline_source_auroc` | `float` | Source probe on source test data |
| `baseline_target_auroc` | `float` | Target probe on target test data (in-arch reference) |
| `transfer_auroc` | `float` | Source probe via `M` applied to target test data |
| `transfer_drop` | `float` | `baseline_target_auroc - transfer_auroc` |

### `learn_alignment(src_acts, tgt_acts, ridge=1e-3) -> torch.Tensor`

Ridge regression. Given paired pooled activations `src_acts ∈ R^{N×d_src}` and `tgt_acts ∈ R^{N×d_tgt}`, learns `M ∈ R^{d_src × d_tgt}` so that `src ≈ M @ tgt` (per-row: `src.T ≈ M @ tgt.T`). Closed form: `M.T = (X.T X + λI)^{-1} X.T Y` where `X = tgt`, `Y = src`. The mapping goes **from target space onto source space** — apply `M` to a target activation to get a source-space approximation. To move a source probe direction `w_src` into target space, use `M.T @ w_src` (this is what `transfer_probe` does).

### `transfer_probe(probe_weights_source, probe_bias_source, alignment) -> (w_target, b_target)`

Given source probe `(w_src, b_src)` and alignment matrix `M` (from `learn_alignment`), returns `w_target = M.T @ w_src` and the unchanged `b_target = b_src`. Resulting probe is `score(x_tgt) = w_target · x_tgt + b_target`.

### `evaluate_transfer(source_model, target_model, source_backend, target_backend, source_tokenize, target_tokenize, align_texts, train_pos, train_neg, test_pos, test_neg, source_layer, target_layer, source_arch_name="source", target_arch_name="target") -> TransferResult`

Full source→target pipeline:

1. Pool source and target activations on `align_texts` (mean over seq dim if 3-D) → fit `M` via `learn_alignment`.
2. Train source probe via `probes.fit_probe` on `(train_pos, train_neg)`. Raises `NotImplementedError` if the source probe head is not `nn.Linear`.
3. Train target probe via `probes.fit_probe` on the same texts (in-arch baseline).
4. Compute `w_transferred = M.T @ w_src` via `transfer_probe`.
5. Evaluate all three settings on `(test_pos, test_neg)`: source probe on source test data, target probe on target test data, transferred probe on target test data. AUROCs go into the returned `TransferResult`.

### `auroc_from_scores(scores, labels) -> float`

Calls `sklearn.metrics.roc_auc_score`. Returns `float("nan")` on `ValueError` (e.g., single-class label vector).

```python
import archscope as ai

src_model, src_tok, src_backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
tgt_model, tgt_tok, tgt_backend = ai.load_model("state-spaces/mamba-130m-hf", arch="mamba")

src_tokenize = ai.make_tokenize_fn(src_tok)
tgt_tokenize = ai.make_tokenize_fn(tgt_tok)

result = ai.transfer.evaluate_transfer(
    src_model, tgt_model, src_backend, tgt_backend,
    src_tokenize, tgt_tokenize,
    align_texts=["text "+str(i) for i in range(32)],
    train_pos=["good "+str(i) for i in range(8)],
    train_neg=["bad "+str(i) for i in range(8)],
    test_pos=["great "+str(i) for i in range(4)],
    test_neg=["awful "+str(i) for i in range(4)],
    source_layer="layer_5.residual",
    target_layer="layer_5.residual",
    source_arch_name="transformer",
    target_arch_name="mamba",
)
print(result.transfer_drop, result.transfer_auroc)
```

---

## archscope.bench

### `InterpProfile` (dataclass)

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Identifier for the report |
| `arch_family` | `str` | `"transformer"` / `"hybrid"` / `"ssm"` / `"custom"` |
| `n_params` | `int` | `sum(p.numel() for p in model.parameters())` |
| `n_layers` | `int` | Number of residual layers |
| `hidden_dim` | `int` | `backend.hidden_dim` of first residual layer |
| `probe_sentiment_auroc` | `float` | Probe AUROC at `n_layers // 4` on built-in sentiment pos/neg |
| `probe_math_auroc` | `float` | Probe AUROC at `n_layers // 2` on built-in math vs non-math |
| `induction_head_relative` | `float` | `induction_head.relative` (x chance baseline) |
| `copy_accuracy` | `float` | `copy_circuit.score` |
| `concentration_relative` | `float` | `early_token_concentration.relative` (0 = concentrated, 1 = uniform) |
| `sae_dense_recon` | `float` | Final-batch recon MSE of `DenseSAE` at `n_layers // 2` |
| `sae_rank1_recon` | `float` | Final-batch recon MSE of `Rank1FactoredSAE` at `n_layers // 2` |
| `sae_better` | `str` | `"dense"` or `"rank1"` based on lower recon |
| `ssm_state_variance_ratio` | `float` | (Mamba only) `var_across_inputs / var_total` of SSM state |
| `runtime_seconds` | `float` | Wall clock |
| `notes` | `list` | Captured per-task error messages (truncated to ~60 chars) |

Every numeric field defaults to `nan` so a partial run still serializes cleanly. Failures inside each test append to `notes` rather than propagating.

### `benchmark(model_name, model, tokenizer, backend_hint, arch_family="transformer", tokenize_fn=None, sentiment_layer=None, math_layer=None, sae_layer=None, ssm_layer=None) -> InterpProfile`

Runs the full suite. Layer choices default to `n_blocks // 4` for sentiment, `n_blocks // 2` for math and SAE; pass explicit ints to override. `ssm_layer` is only used when `arch_family == "ssm"`. The function never raises — every test is in its own `try/except` that pushes to `notes`.

### `profile_to_markdown(profile) -> str`

Renders a markdown block with header (arch / params / layers / hidden), a small results table, and any captured notes.

```python
import archscope as ai

model, tok, _ = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
tokenize = ai.make_tokenize_fn(tok)

profile = ai.bench.benchmark(
    model_name="pythia-160m", model=model, tokenizer=tok,
    backend_hint="transformer", arch_family="transformer",
    tokenize_fn=tokenize,
)
print(ai.bench.profile_to_markdown(profile))
```

---

## CLI

Installed entry point: `archscope` (declared in `pyproject.toml` as `archscope.cli:cli`).

### `archscope info`

Prints two `rich.Table`s:

1. Method → module path → source paper, for: Probes, SAE, Neuron mod, Activation patch, Cross-arch transfer, Circuit detection, Logit/tuned lens, Model diff, InterpBench.
2. Backend name → architecture family covered, for: `transformer`, `mamba`, `kazdov`, `recurrent`.

### `archscope bench MODEL_NAME --arch ARCH [--out PATH]`

Runs `bench.benchmark` on a HuggingFace model.

- `MODEL_NAME` — HF model id (positional).
- `--arch` — one of `transformer` / `mamba` / `kazdov` (defaults to `transformer`). Mapped to `arch_family`: transformer→transformer, mamba→ssm, kazdov→hybrid.
- `--out` — output file. Without it, the markdown report is printed to stdout. With it:
  - `.md` extension → markdown.
  - `.json` extension (or no extension) → `dataclasses.asdict(profile)` via `json.dump(indent=2, default=str)`.
  - Anything else → `click.UsageError`.

When `--arch mamba` is passed, the CLI also auto-picks `ssm_layer = n_residual_layers // 2` so the `ssm_state_variance_ratio` field is populated (otherwise it stays NaN).

```
$ archscope info
$ archscope bench EleutherAI/pythia-160m --arch transformer
$ archscope bench EleutherAI/pythia-160m --arch transformer --out pythia.md
$ archscope bench state-spaces/mamba-130m-hf --arch mamba --out mamba.json
```
