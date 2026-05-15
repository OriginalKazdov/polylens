# Tutorial: your first archscope experiment

## 1. What you'll do

Train a sentiment probe on Pythia-160m, extract Mamba-130m's recurrent SSM state,
run a logit lens, and finish with the headline experiment: take the Pythia probe
direction, align Mamba's activation space to Pythia's, and check whether the
direction still classifies sentiment on Mamba. End-to-end on CPU in a few minutes.

This is `archscope` v0.2.7. Every code block runs if pasted in order.

---

## 2. Install

```bash
pip install archscope
```

archscope depends on `transformers`, `torch`, and `scikit-learn` (pulled in automatically).

For Mamba models, `transformers` already ships an HF-pure Mamba implementation that
runs on CPU — `mamba-ssm` (the CUDA kernel package) is **optional** and only helps
on GPU. The tutorial below works on CPU without it.

---

## 3. Load a model

```python
import archscope as ai

model, tok, backend = ai.load_model(
    "EleutherAI/pythia-160m",
    arch="transformer",
)
```

`ai.load_model` returns a 3-tuple:

- **model**: the HuggingFace `AutoModelForCausalLM` (`.eval()` already called, `dtype=float32`).
- **tok**: the matching `AutoTokenizer` with `pad_token` set to `eos_token` if it was `None`.
- **backend**: an `archscope.backends.Backend` — the object that knows how to pull
  named activations out of *this particular architecture*. You'll use it directly
  in step 4.

`arch="..."` is a backend hint. If you omit it, archscope auto-detects from
`model.config.model_type` (Pythia is `gpt_neox` → `transformer`; Mamba is
`mamba` → `mamba`). Pass `arch` explicitly when (a) you want to be defensive, or
(b) the model type isn't in the autodetect table — see `Backend._AUTODETECT` for
the current list.

---

## 4. Extract your first activation

A layer name in archscope is a string like `"layer_5.residual"` (transformer) or
`"layer_12.ssm_state"` (mamba). To see them all:

```python
backend.layer_names()[:3]
# ['layer_0.residual', 'layer_1.residual', 'layer_2.residual']
```

For Pythia-160m there are 12 such names. Now extract layer 5's residual stream:

```python
inputs = tok(["The cat sat on the mat."], return_tensors="pt")
records = backend.extract(inputs, layers=["layer_5.residual"])
rec = records[0]

print(rec.activations.shape)   # torch.Size([1, 8, 768])
print(rec.meta)                # {'kind': 'residual', 'arch': 'transformer'}
```

The activation shape is **(batch, seq_len, hidden_dim)**: one example, 8 tokens,
768-dim residual stream. `rec.meta` carries arch-specific info — useful when you
mix architectures and need to remember what kind of state you grabbed.

If you ask for a name that doesn't exist, `backend.extract` raises a `ValueError`
that includes the first few valid layer names. No silent failure.

---

## 5. Train a probe

`archscope.probes.fit_probe` does the whole pipeline: extract activations at the
named layer, pool over sequence, fit a linear (or MLP) probe, return metrics.

Build a tiny dataset (5 pos + 5 neg — enough to see the API; not enough for
believable AUROC):

```python
pos = ["I love this", "Amazing show", "Wonderful day",
       "Fantastic work", "Truly delightful"]
neg = ["I hate this", "Awful film", "Terrible day",
       "Disappointing work", "Truly dreadful"]

pf = ai.probes.fit_probe(
    model,
    tokenizer=tok,
    pos_texts=pos, neg_texts=neg,
    layer_name="layer_7.residual",
    backend_hint="transformer",
)
print(pf.metrics)
# {'train_auroc': 1.0, 'val_auroc': 0.5, 'train_loss': 0.31}
```

With only 10 examples and a clean lexical split, the train probe trivially fits
(`train_auroc=1.0`). `val_auroc=0.5` here is the documented "only one class in
the val split" edge case — at 10 examples, the 20% val split is two examples and
they can both end up the same label. Archscope emits a warning and returns
`0.5` rather than NaN. For a real result you want ~50+ examples per class with
subtler phrasings (see step 8).

`pf` is a `ProbeFit`. Two attributes matter beyond `.metrics`:

```python
pf.direction        # torch.Size([768]) — the probe's weight vector in
                    # activation space. A unit-ish direction along which positive
                    # > negative in this layer's residual stream.
pf.bias             # torch.Size([])   — the scalar offset.
```

For any activation `a` (a 768-vector at this layer) the probe's logit is exactly
`a @ pf.direction + pf.bias`. Probability is `sigmoid(...)`. This decoupling is
load-bearing for step 8.

---

## 6. Switch to Mamba and grab the SSM state

```python
mamba_model, mamba_tok, mamba_backend = ai.load_model(
    "state-spaces/mamba-130m-hf",
    arch="mamba",
)
```

Mamba exposes two kinds of state per block:

```python
mamba_backend.layer_names()[:4]
# ['layer_0.residual', 'layer_0.ssm_state',
#  'layer_1.residual', 'layer_1.ssm_state']
```

`.residual` is the residual stream (same shape semantics as a transformer).
`.ssm_state` is the **final recurrent state `h_T`** after the block has processed
the whole sequence — Mamba's analog of an RNN's last hidden state.

```python
m_inputs = mamba_tok(["The cat sat on the mat."], return_tensors="pt")
rec = mamba_backend.extract(m_inputs, layers=["layer_12.ssm_state"])[0]

print(rec.activations.shape)   # torch.Size([1, 1536, 16])
print(rec.meta["shape_meaning"])
# '(B, intermediate_size, ssm_state_size)'
```

The SSM state is **(B, intermediate_size, ssm_state_size)** = `(1, 1536, 16)`
for `mamba-130m-hf`. Note the missing sequence axis: this is *one* state vector
per example, summarising everything the block has read so far. That's the whole
reason Mamba is interesting — recurrence is explicit, and you can grab `h_T`
without re-running anything. Treat it as a `(1536 × 16) = 24,576`-dim feature
vector if you want to feed it to a probe or SAE.

---

## 7. Run a logit lens

The logit lens (Nostalgebraist 2020) takes each layer's residual stream, applies
the model's *own* final norm and unembedding, and reads off "what would the model
predict if forced to commit at this layer?"

```python
result = ai.lens.logit_lens(
    model, tok,                       # the Pythia-160m from step 3
    prompt="The capital of France is",
    target_token=" Paris",
    backend_hint="transformer",
)
print(result.to_markdown())
```

Actual output on Pythia-160m (12 layers):

```
### logit_lens on `The capital of France is`
Target: ` Paris` (id=7785)

| Layer | top-1 token  | top-1 prob | target prob |  rank | entropy |
|------:|--------------|-----------:|------------:|------:|--------:|
|     0 | ` always`    |      0.604 |       0.000 |  5117 |   1.638 |
|     1 | ` now`       |      0.434 |       0.000 |  8622 |   1.702 |
|     4 | ` now`       |      0.519 |       0.000 | 14235 |   1.435 |
|     7 | ` currently` |      0.467 |       0.000 | 11612 |   2.682 |
|     9 | ` located`   |      0.492 |       0.000 |   466 |   2.046 |
|    10 | ` located`   |      0.541 |       0.000 |   361 |   1.188 |
|    11 | ` in`        |      0.067 |       0.001 |    77 |   7.249 |
```

Read this as a trajectory: target rank drops from ~5000 → 77 by the final
layer as the model's internal computation moves toward the answer, even though
Pythia-160m is too small to surface ` Paris` as top-1. Entropy spikes at the
final layer because Pythia is calibrating "in / located / inhabited / …" rather
than committing to a city name.

**Caveat on Mamba.** A naive logit lens degrades with depth on Mamba because the
residual stream isn't trained to be unembed-decodable at every layer. If you
care about deep-layer readouts on Mamba, fit a `TunedLens` (Belrose et al 2023)
instead — `ai.lens.TunedLens.fit(model, tok, calibration_texts, backend_hint="mamba")`
learns per-layer affine corrections.

---

## 8. Cross-architecture transfer (the main wedge)

This is the experiment archscope is built for. Train a probe on Pythia. Learn a
linear map from Mamba's activation space into Pythia's. Apply the original
Pythia direction to *Mamba activations expressed in Pythia coordinates* and see
how much of the sentiment signal survives.

**Walkthrough.** Use the same texts to anchor both models — that's what
"paired activations" means.

```python
import torch
from archscope.transfer import learn_alignment

# Pretend these are your real splits — see examples/cross_arch_sentiment_transfer.py
# for the 45+45 dataset with subtle/mixed cases.
pair_texts = pos + neg                          # any texts; no labels needed for align
test_texts = ["Wonderful day", "Awful film"]    # one pos, one neg
```

Step 1 — extract paired activations on the **same texts** from both models, at
each model's chosen layer:

```python
def pool(backend, tokr, texts, layer):
    rec = backend.extract(tokr(texts, return_tensors="pt", padding=True), layers=[layer])[0]
    return rec.activations.mean(dim=1).detach()    # (N, hidden)

src_paired = pool(backend,        tok,       pair_texts, "layer_7.residual")
tgt_paired = pool(mamba_backend,  mamba_tok, pair_texts, "layer_23.residual")
```

Step 2 — learn the alignment `M` (ridge regression, target → source space):

```python
M = learn_alignment(src_paired, tgt_paired, ridge=0.001)
print(M.shape)    # torch.Size([768, 768])
                  # (d_src=Pythia hidden, d_tgt=Mamba hidden_size; equal here)
```

Step 3 — extract Mamba test activations, project them into Pythia space:

```python
tgt_test = pool(mamba_backend, mamba_tok, test_texts, "layer_23.residual")
tgt_test_in_src_space = tgt_test @ M.T              # (N, d_src)
```

Step 4 — apply the source probe direction *manually* (this is why we exposed
`pf.direction` / `pf.bias`):

```python
logits = tgt_test_in_src_space @ pf.direction + pf.bias    # (N,)
probs = torch.sigmoid(logits)
print(probs)
```

The probe was never trained on a single Mamba activation. Whatever AUROC you
get is purely the geometric overlap between the two activation spaces.

**Real numbers** from the proper experiment (45 pos + 45 neg, 80/20 split, 3
seeds) in `examples/cross_arch_sentiment_transfer.py`:

| Probe | In-arch AUROC | Cross-arch (transferred) | Drop |
|---|---:|---:|---:|
| Pythia layer 7 → Mamba layer 23 | 0.667 | 0.605 | 0.062 |
| Mamba layer 23 → Pythia layer 7 | 0.872 | 0.704 | 0.168 |

Both transfers stay clearly above chance (0.500). The Pythia → Mamba direction
loses very little (Pythia's own probe was already weak); Mamba → Pythia loses
~17 AUROC points but is still well above chance, suggesting a sentiment direction
that's *partially* shared across architectures up to a linear map.

These numbers are small-model, small-dataset — the point is the methodology.
Scaling the dataset and re-running across more architectures is exactly what
archscope is designed to make cheap. Full reproducible script at
`examples/cross_arch_sentiment_transfer.py`.

---

## 9. What archscope is NOT

archscope is not `transformer_lens` — that's the broader, more mature
transformer-only toolkit and you should use it for serious transformer-only
work (deeper hook surface, more circuits, much larger user base). archscope is
not `nnsight` either — nnsight is a more general intervention/proxy framework
across PyTorch models. archscope is also not for production audit, not a SAE
training rig at scale, and not a serving framework. It's a small workbench for
running cross-arch experiments on small (≤1B parameter) models — the kind of
thing where you want one library that gives you comparable activations from a
transformer, a Mamba, and a custom recurrent model without writing three
different hook setups.

---

## 10. Next steps

- `docs/cookbook.md` — recipes: SAEs on SSM state, induction-head detection,
  base-vs-finetune diff, the full InterpProfile benchmark.
- `docs/api.md` — module-by-module reference.
- `examples/cross_arch_sentiment_transfer.py` — the script that produced the
  AUROC table in step 8.
- `examples/quickstart.py` — four short demos in one file.
