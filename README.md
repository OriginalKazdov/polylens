# polylens

**Mechanistic interpretability across architectures — one API for transformers, SSMs (Mamba), and hybrid models.**

[![CI](https://github.com/OriginalKazdov/polylens/actions/workflows/ci.yml/badge.svg)](https://github.com/OriginalKazdov/polylens/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Existing mech interp tools (`transformer_lens`, `nnsight`, `sae_lens`) are transformer-only. `polylens` works across architectures with a single API, including **the first open-source Mamba SSM-state extraction**.

```python
import polylens as mi
from transformers import AutoModelForCausalLM, AutoTokenizer

tok   = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

backend = mi.backends.Backend.for_model(model, hint="mamba")

# Extract Mamba's recurrent SSM state h_t — unique to polylens
ssm = backend.extract(tok("text", return_tensors="pt"), layers=["layer_12.ssm_state"])[0]
# Shape: (B, intermediate_size, ssm_state_size) = (B, 1536, 16) for mamba-130m
```

---

## Why this exists

Mech interp tools assume transformers. As Mamba, hybrid attention, and custom architectures proliferate, this gap matters: **we can't compare what we can't probe**. `polylens` is the library that probes them all under one API.

---

## What's inside (v0.2.0)

### 7 mech interp methods, one API

| Module | What it does | Source |
|---|---|---|
| `probes` | Linear/MLP probes on hidden states | Drop the Act (arXiv:2605.11467) |
| `sae` | Dense + Rank-1 factored sparse autoencoders | WriteSAE (arXiv:2605.12770) |
| `neurons` | Top-K contrastive neuron modulation | Targeted Neuron Mod (arXiv:2605.12290) |
| `attribute` | Activation patching + DIM decomposition | Multi-Agent Sycophancy (arXiv:2605.12991) |
| `circuits` | Induction head, copy, attention-sink detectors | Olsson et al 2022 |
| `lens` | Logit lens + Tuned lens (Belrose et al 2023) | this library |
| `diff` | Model-diff: base vs fine-tuned, find what changed | this library |
| `transfer` | Cross-arch probe transfer via linear alignment | this library |
| `bench` | InterpProfile standardized profile | this library |

### 4 backends, one API

| Backend | Models | Unique capability |
|---|---|---|
| `transformer` | Pythia, GPT-2, Llama, Mistral, Qwen, MPT, Falcon, GPT-Neo | residual stream |
| `mamba` | Mamba, Mamba-2 | residual + **`.ssm_state`** (h_t) |
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

For Mamba support on CPU you don't need `mamba-ssm` — HF's slow path works. On CUDA install `mamba-ssm` for the fast path.

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

### Extract Mamba's SSM internal state (unique)

```python
backend = mi.backends.Backend.for_model(mamba_model, hint="mamba")
rec = backend.extract(tk("Hello world"), layers=["layer_12.ssm_state"])[0]
# rec.activations.shape == (B, intermediate_size, ssm_state_size)
# This is the actual recurrent memory of Mamba — the SSM h_t at end of sequence.
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

## Findings — what we saw running polylens on 7 small models

Selected results from running `bench.benchmark()` across model families. Full table + JSON in `_research/mini_zoo_leaderboard.json`.

| Model | Arch | Params | Induction (× chance) | SAE rank-1 vs dense | Notes |
|---|---|---|---|---|---|
| Pythia-160m | transformer | 162M | **490×** | dense better | baseline |
| Pythia-410m | transformer | 405M | **3261×** | dense better | 6.6× induction at 2.5× params |
| GPT-2 | transformer | 124M | **6393×** | **rank-1 ~10× better** | older training, recurrent-friendly residuals? |
| Mamba-130m | SSM | 130M | TBD (running) | TBD | + SSM-state extraction |
| Mamba-370m | SSM | 370M | TBD | TBD | scale check |
| Qwen2.5-0.5B | transformer | 500M | TBD | TBD | modern transformer |
| kazdov-α | hybrid | 98M | TBD | **rank-1 dominates** | math-pretrained MoBE-BCN+MHA |

**Cross-architecture observations** (some preliminary, full study in upcoming blog):

- **Induction-head behavior is not transformer-exclusive**. Mamba showed 6378× chance on a related test in our earlier 3-model run (paper-tier finding worth confirming at scale).
- **Logit lens fails on Mamba**. Pythia logit-lens correctly surfaces target token (rank 5117 → 77 across 12 layers). Mamba logit-lens *degrades* with depth (rank 197 → 1049). Mamba's intermediate residuals are not in vocab-space — tuned-lens is motivated.
- **Rank-1 SAE preference is layer- and architecture-dependent**, not a clean architectural property. GPT-2 strongly prefers rank-1; Pythia prefers dense; kazdov shows extreme local rank-1 wins at specific layers.

---

## Honest limits

`polylens` is a v0.2 release. What it does well: cross-architecture mech interp primitives, unified API, real findings, validated on 3+ architectures. What it doesn't do (yet):

- No causal scrubbing (gold-standard verification)
- No interactive notebook viz (matplotlib helpers TBD)
- No multi-token circuit detection (IOI, name-mover) — only induction/copy/concentration
- Mamba-2 backend support is partial
- Pretrained SAE collection isn't shipped (you train your own per layer)
- Probe transfer requires same-tokenizer paired data

See `CONTRIBUTING.md` for what we welcome (new backends, new circuit detectors, viz helpers).

---

## Citation

```bibtex
@misc{dovzak2026polylens,
  title  = {polylens: A cross-architecture mechanistic interpretability toolkit},
  author = {Juan Cruz Dovzak},
  year   = {2026},
  url    = {https://github.com/OriginalKazdov/polylens}
}
```

Papers reimplemented or wrapped:
- WriteSAE — arXiv:2605.12770
- Drop the Act / ProFIL — arXiv:2605.11467
- Targeted Neuron Modulation — arXiv:2605.12290
- Multi-Agent Sycophancy — arXiv:2605.12991
- Tuned Lens (Belrose et al, 2023)
- Induction heads (Olsson et al, 2022)

---

## Troubleshooting

### "The fast path is not available because ..." (Mamba on CPU)

Normal. Mamba falls back to a slow pure-PyTorch path that works correctly (~30s per benchmark instead of ~1s). Install `pip install mamba-ssm causal-conv1d` only on CUDA machines.

### Custom backend not auto-detected

Pass `Backend.for_model(model, hint="my_backend")` explicitly. Auto-detection uses `config.model_type`.

### `RuntimeError: Trying to backward through the graph a second time`

Activations from `Backend.extract()` carry the autograd graph by default. Call `.detach()` before reusing, or extract inside `torch.no_grad()`. The high-level `probes.fit_probe()` does this for you.

---

## Roadmap (post-0.2.0)

- Multi-token circuit detection: IOI, name-mover, successor heads
- Mamba-2 backend (architecture differs from Mamba-1)
- Cross-arch SAE feature alignment (extend `transfer.py` from probes to features)
- Pretrained SAE collection for common small models
- Plotly/matplotlib viz helpers
- HuggingFace Space demo

PRs welcome — see `CONTRIBUTING.md`.

---

## License

Apache-2.0
