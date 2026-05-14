# mechinterp-small

**Unified mechanistic interpretability toolkit for small models across architectures — transformers, hybrid attention, and state-space models (Mamba).**

Existing mech interp libraries (`transformer_lens`, `nnsight`, `sae_lens`) are transformer-only. `mechinterp-small` works across architectures with a single API, including **the first open-source Mamba SSM state extraction**.

```python
import mechinterp_small as mi
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
backend = mi.backends.Backend.for_model(model, hint="mamba")

# Extract Mamba's internal SSM state — unique among mech interp libraries
records = backend.extract(inputs, layers=["layer_12.ssm_state"])
# Shape: (batch, intermediate_size, ssm_state_size) — the actual recurrent memory
```

## Why this exists

Mech interp tools assume transformer architecture. As Mamba and hybrid models proliferate, this gap matters: **we can't compare what we can't probe**. `mechinterp-small` is the library that probes them all.

## Features

- ✅ **4 mech interp primitives** unified under a single API:
  - **Linear probes** over residual stream (Drop the Act-style, arXiv:2605.11467)
  - **Sparse autoencoders** (Dense + Rank-1 factored, WriteSAE-style, arXiv:2605.12770)
  - **Targeted neuron modulation** (arXiv:2605.12290)
  - **Activation patching + DIM decomposition** (arXiv:2605.12991)

- ✅ **3 backends, one API**:
  - `transformer` — HuggingFace decoder LMs (Pythia, Llama, GPT, Qwen)
  - `mamba` — Mamba / Mamba-2 SSMs, **including .ssm_state extraction**
  - `kazdov` — Hybrid MoBE-BCN+MHA (custom architectures registerable)

- ✅ **Cross-architecture experiments**:
  - **Probe transfer** between archs via paired-activation linear alignment
  - **Circuit detection** (induction head, copy, attention concentration) — works on any arch
  - **InterpBench** standardized benchmark with leaderboard JSON output

## Install

```bash
pip install mechinterp-small
```

For Mamba support, you don't need `mamba-ssm` (we use HuggingFace's slow path which works on CPU).

## Quick start

### Train a probe on any architecture

```python
import mechinterp_small as mi
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

def tokenize(texts):
    return tok(texts, return_tensors="pt", padding=True, truncation=True)

probe = mi.probes.fit_probe(
    model,
    inputs_pos=tokenize(["I love this", "Wonderful!", "Amazing show"]),
    inputs_neg=tokenize(["I hate this", "Awful day", "Terrible meal"]),
    layer_name="layer_5.residual",
    backend_hint="transformer",
)
print(probe.metrics)   # {'train_auroc': 1.0, 'val_auroc': 0.95, 'train_loss': 0.3}
```

### Extract Mamba's SSM state (unique feature)

```python
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
backend = mi.backends.Backend.for_model(model, hint="mamba")

# Standard residual stream
residual = backend.extract(inputs, layers=["layer_12.residual"])[0]
# Shape: (B, T, hidden_size)

# Mamba SSM internal state — recurrent "memory"
ssm = backend.extract(inputs, layers=["layer_12.ssm_state"])[0]
# Shape: (B, intermediate_size, ssm_state_size)  = (B, 1536, 16)
```

### Run InterpBench on any model

```python
profile = mi.bench.benchmark(
    "EleutherAI/pythia-160m", model, tokenizer,
    backend_hint="transformer", arch_family="transformer",
    tokenize_fn=tokenize,
)
print(mi.bench.profile_to_markdown(profile))
```

CLI:
```bash
mechinterp bench EleutherAI/pythia-160m --arch transformer --out pythia.json
mechinterp bench state-spaces/mamba-130m-hf --arch mamba
```

### Detect circuits across architectures

```python
scores = mi.circuits.run_all_circuits(model, tokenizer=tok)
print(scores["induction_head"].relative)   # e.g., 6378.5 (Mamba)  vs  490.2 (Pythia)
print(scores["copy_circuit"].score)        # accuracy
print(scores["early_token_concentration"].relative)
```

### Cross-architecture probe transfer

```python
from mechinterp_small.transfer import evaluate_transfer

result = evaluate_transfer(
    source_model=pythia,  target_model=mamba,
    source_backend=pythia_backend, target_backend=mamba_backend,
    source_tokenize=pythia_tokenize, target_tokenize=mamba_tokenize,
    align_texts=ALIGN_TEXTS,
    train_pos=MATH_TRAIN, train_neg=NONMATH_TRAIN,
    test_pos=MATH_TEST,   test_neg=NONMATH_TEST,
    source_layer="layer_5.residual", target_layer="layer_12.residual",
)
print(f"Transfer AUROC: {result.transfer_auroc} (drop: {result.transfer_drop:+.3f})")
```

## What we found running this on 3 models

| Model | Arch | Sentiment | Math | Induction | Copy | Concentration | SAE-Dense | SAE-Rank1 | SSM-Var |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-160m | transformer | 0.99 | 1.00 | 490× | 0.0% | 0.57 | 0.020 | 0.028 | — |
| Mamba-130m | SSM | 1.00 | 1.00 | **6378×** | 0.0% | 0.51 | 0.031 | 0.046 | 0.535 |
| kazdov-α-98m | hybrid | 0.95 | 1.00 | 2700× | 3.3% | 0.36 | 0.362 | **0.005** | — |

**Cross-architecture findings**:
- **Mamba develops induction-like behavior 13× stronger than parameter-matched Pythia**, despite having no attention mechanism.
- **Hybrid attention (kazdov) is dramatically better-served by Rank-1 SAEs** (~70× lower recon at mid-layer) than by standard dense SAEs.
- **Math/sentiment probes transfer cleanly across all 3 architectures** at mid-layers, suggesting some semantic features are arch-agnostic.

## Architecture coverage

| Backend | Arch family | Extraction support | Special |
|---|---|---|---|
| `transformer` | Decoder LMs (Llama, GPT, Pythia, Qwen, ...) | Residual stream | — |
| `mamba` | Mamba, Mamba-2 SSM | Residual + **SSM recurrent state** | UNIQUE — exposes h_t for the first time |
| `kazdov` | Hybrid MoBE-BCN+MHA | Residual after each block | Custom block class registration |
| `recurrent` | Generic RNN (user subclass) | Hidden state per layer | Override `extract()` in subclass |

## Citation

If you use this library or its benchmark, please cite:
```
@misc{dovzak2026mechinterpsmall,
  title  = {mechinterp-small: A cross-architecture mechanistic interpretability toolkit},
  author = {Juan Cruz Dovzak},
  year   = {2026},
  url    = {https://github.com/kazdov/mechinterp-small}
}
```

Source papers reimplemented or wrapped:
- WriteSAE — arXiv:2605.12770
- Drop the Act / ProFIL — arXiv:2605.11467
- Targeted Neuron Modulation — arXiv:2605.12290
- Multi-Agent Sycophancy — arXiv:2605.12991

## License

Apache-2.0
