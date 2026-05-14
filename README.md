# polylens

**Mechanistic interpretability experiments across architectures — Transformers, SSMs/Mamba, recurrent models, and hybrids.**

[![CI](https://github.com/OriginalKazdov/polylens/actions/workflows/ci.yml/badge.svg)](https://github.com/OriginalKazdov/polylens/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Existing mech-interp tools are strongest around Transformer-centric workflows. `polylens` focuses on **comparable experiments across Transformers, SSMs/Mamba, recurrent models, and custom hybrid architectures** under one lightweight API.

The differentiator isn't replacing `transformer_lens` or `nnsight` (both excellent and broader). It's making **cross-architecture experiments easier**: probes, SAE comparisons, circuit-style metrics, logit/tuned lens, model diffs, and explicit Mamba recurrent-state extraction in one place.

```python
import polylens as mi
from transformers import AutoModelForCausalLM, AutoTokenizer

tok   = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

backend = mi.backends.Backend.for_model(model, hint="mamba")

# Extract Mamba's recurrent SSM state h_t (in addition to residual stream)
ssm = backend.extract(tok("text", return_tensors="pt"), layers=["layer_12.ssm_state"])[0]
# Shape: (B, intermediate_size, ssm_state_size) = (B, 1536, 16) for mamba-130m
```

---

## What's inside (v0.2.0)

### Core mech-interp methods

| Module | What it does | Source |
|---|---|---|
| `probes` | Linear/MLP probes on hidden states | Drop the Act (arXiv:2605.11467) |
| `sae` | Dense + Rank-1 factored sparse autoencoders | WriteSAE (arXiv:2605.12770) |
| `neurons` | Top-K contrastive neuron modulation | Targeted Neuron Mod (arXiv:2605.12290) |
| `attribute` | Activation patching + DIM decomposition | Multi-Agent Sycophancy (arXiv:2605.12991) |
| `circuits` | Induction, copy, attention-concentration detectors | Olsson et al 2022 |
| `lens` | Logit lens + Tuned lens | Belrose et al 2023 |
| `diff` | Model-diff: base vs fine-tuned, find what changed | this library |

### Experiment infrastructure

| Module | What it does |
|---|---|
| `backends` | Unified extraction API across architectures |
| `transfer` | Cross-arch probe transfer via paired-activation linear alignment |
| `bench` | InterpProfile — standardized comparable profile (`mi.bench.benchmark()`) |

### Backends

| Backend | Models | Specific |
|---|---|---|
| `transformer` | Pythia, GPT-2, Llama, Mistral, Qwen, MPT, Falcon, GPT-Neo | residual stream |
| `mamba` | Mamba, Mamba-2 | residual + explicit `.ssm_state` (recurrent h_t) |
| `kazdov` | Kazdov-α hybrid MoBE-BCN+MHA | residual per custom block |
| `recurrent` | Generic RNN (user subclass) | hidden state per layer |

---

## Install

```bash
pip install polylens   # once on PyPI
# or:
git clone https://github.com/OriginalKazdov/polylens.git
cd polylens && pip install -e .
```

For Mamba on CPU you don't need `mamba-ssm` — HF's slow path works. On CUDA install `mamba-ssm` for the fast path.

---

## Quick examples

### Train a probe on any architecture

```python
import polylens as mi
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
tok   = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
tk = lambda txts: tok(txts, return_tensors="pt", padding=True, truncation=True)

probe = mi.probes.fit_probe(
    model,
    inputs_pos=tk(["I love this", "Wonderful!", "Amazing"]),
    inputs_neg=tk(["I hate this", "Awful", "Terrible"]),
    layer_name="layer_5.residual",
    backend_hint="transformer",
)
print(probe.metrics)   # {'train_auroc': 1.0, ...}
```

### Extract Mamba's SSM recurrent state

```python
backend = mi.backends.Backend.for_model(mamba_model, hint="mamba")
rec = backend.extract(tk("Hello world"), layers=["layer_12.ssm_state"])[0]
# rec.activations.shape == (B, intermediate_size, ssm_state_size)
# This is the actual recurrent memory h_t of Mamba — exposed via the same
# extraction API used for Transformer residual streams.
```

### Logit lens / tuned lens — see what each layer "thinks"

```python
result = mi.lens.logit_lens(
    model, tok,
    prompt="The capital of France is",
    target_token=" Paris",
    backend_hint="transformer",
)
print(result.to_markdown())

# Tuned lens — learned per-layer projections (Belrose et al 2023):
tl = mi.lens.TunedLens.fit(model, tok, calibration_texts, backend_hint="transformer")
tl.predict(model, tok, "...", backend_hint="transformer")
```

### Model Diff — what did fine-tuning change?

```python
from polylens.diff import compare

result = compare(
    base_model, fine_tuned_model, tokenizer,
    calibration_texts=texts,
    backend_hint="transformer",
)
print(result.to_markdown())
# Per-layer residual drift, top shifted neurons, circuit deltas.
```

### Detect circuits cross-arch

```python
scores = mi.circuits.run_all_circuits(model, tokenizer=tok)
print(scores["induction_head"].relative)   # × chance
print(scores["copy_circuit"].score)        # accuracy
```

### InterpBench — standardized model profile

```python
profile = mi.bench.benchmark(
    "EleutherAI/pythia-160m", model, tok,
    backend_hint="transformer", arch_family="transformer",
    tokenize_fn=tk,
)
print(mi.bench.profile_to_markdown(profile))
```

CLI:
```bash
polylens info
polylens bench EleutherAI/pythia-160m --arch transformer --out pythia.json
polylens bench state-spaces/mamba-130m-hf --arch mamba
```

---

## Findings — running polylens on a mini-zoo of 7 small models

Each model profiled with `bench.benchmark()` (probes + circuits + dense vs rank-1 SAE). Full JSON in `_research/mini_zoo_leaderboard.json`. ~10 min total compute on CPU.

| Model | Arch | Params | Induction (× chance) | SAE-dense | SAE-rank1 | SSM var |
|---|---|---|---|---|---|---|
| Pythia-160m | transformer | 162M | 490× | 0.019 | 0.025 | — |
| Pythia-410m | transformer | 405M | 3,261× | 0.075 | 0.135 | — |
| GPT-2 | transformer | 124M | 6,393× | 5.731 | **0.608** | — |
| Mamba-130m | SSM | 129M | 6,378× | 0.048 | **0.032** | 0.54 |
| Mamba-370m | SSM | 372M | **7,730×** | 0.022 | 0.027 | 0.73 |
| Qwen2.5-0.5B | transformer | 494M | **17,637×** | 0.092 | 0.068 | — |
| kazdov-α | hybrid | 98M | 2,700× | 0.043 | **0.004** | — |

**Observations** (preliminary — observations from running the library, not formal claims):

- **Induction-like behavior shows up strongly in non-attention architectures.** Mamba — which has no attention mechanism — scores 6378-7730× chance on a synthetic induction-copy test, comparable to or above similarly-sized Transformers. The test itself is behavioral (output-based), so it doesn't presume any particular implementation mechanism. Worth investigating *what* SSMs use to implement this behavior.
- **Naive logit lens appears poorly calibrated on Mamba.** Applying Pythia's `lm_head` to intermediate residuals surfaces target tokens with depth (Pythia rank 5117 → 77 across 12 layers on "capital of France is _Paris_"). The same procedure on Mamba *degrades* the target with depth (rank 197 → 1049). Mamba's intermediate residuals don't seem to live in vocab-aligned space — `TunedLens.fit()` is motivated for SSMs.
- **Rank-1 vs dense SAE preference splits by architecture family.** GPT-2, both Mambas, and kazdov-α all reconstruct better with rank-1 factored SAEs at the tested mid-layer. Both Pythias prefer dense. Qwen is marginal. The pattern is suggestive but not yet conclusive — needs layer sweeps + multiple seeds.
- **Modern transformer training boosts induction strongly.** Qwen2.5-0.5B shows 17,637× induction — 5.4× higher than Pythia-410m at similar size. Likely reflects data curation + training stability improvements since Pythia (2023).
- **Mamba's SSM-state utilization scales with size.** Variance ratio across diverse inputs: 0.54 (130m) → 0.73 (370m). Larger Mamba models encode more input-specific information in their recurrent state.

These aren't published results — they're observations from running the library. We'd happily be corrected if anyone finds methodological issues.

---

## Honest limits

`polylens` is a v0.2 release. What it does well: cross-architecture mech-interp primitives, unified API, real observable findings, validated on multiple architectures. What it doesn't do yet:

- No causal scrubbing (gold-standard circuit verification)
- No interactive notebook viz (matplotlib helpers are TBD)
- Circuit detection is limited to induction / copy / attention-concentration — no IOI, name-mover, or successor heads yet
- Mamba-2 backend support is partial (Mamba-1 fully supported)
- No pretrained SAE collection (you train your own per layer)
- Probe transfer assumes same-tokenizer paired data

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for what we welcome (new backends, new circuit detectors, viz helpers).

For mature transformer-only work prefer [`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens) (much broader feature surface) or [`nnsight`](https://nnsight.net/) (modern PyTorch tracing). They overlap with us on the transformer side; we focus where they're lighter.

---

## Citation

```bibtex
@misc{dovzak2026polylens,
  title  = {polylens: Cross-architecture mechanistic interpretability experiments},
  author = {Juan Cruz Dovzak},
  year   = {2026},
  url    = {https://github.com/OriginalKazdov/polylens}
}
```

Source papers reimplemented or wrapped:
- WriteSAE — arXiv:2605.12770
- Drop the Act / ProFIL — arXiv:2605.11467
- Targeted Neuron Modulation — arXiv:2605.12290
- Multi-Agent Sycophancy — arXiv:2605.12991
- Tuned Lens (Belrose et al, 2023)
- Induction heads (Olsson et al, 2022)

---

## Troubleshooting

### "The fast path is not available because ..." (Mamba on CPU)

Normal. Mamba falls back to a slow pure-PyTorch path that works correctly (~30s per benchmark vs ~1s on CUDA). Install `pip install mamba-ssm causal-conv1d` only on CUDA machines.

### Custom backend not auto-detected

Pass `Backend.for_model(model, hint="my_backend")` explicitly. Auto-detection uses `config.model_type`.

### `RuntimeError: Trying to backward through the graph a second time`

Activations from `Backend.extract()` carry the autograd graph by default. Call `.detach()` before reusing, or extract inside `torch.no_grad()`. The high-level `probes.fit_probe()` does this for you.

---

## Roadmap (post-0.2.0)

- Multi-token circuit detection: IOI, name-mover, successor heads
- Mamba-2 backend with same `.ssm_state` API
- Cross-arch SAE feature alignment (extend `transfer.py` from probes to features)
- Pretrained SAE collection for common small models
- Plotly/matplotlib viz helpers
- HuggingFace Space demo

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

Apache-2.0
