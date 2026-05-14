# Contributing to polylens

Thanks for considering a contribution. Brief guidelines.

## Scope of this library

`polylens` is intentionally focused on:
- **Small models** (typically <2B params, runnable on CPU/laptop GPU)
- **Cross-architecture support** (transformer + SSM + hybrid + custom)
- **Mechanistic interpretability primitives** wrapped under a unified API

We are **not** trying to compete with `transformer_lens` (the canonical transformer-only lib) or `nnsight` (Stanford's modern alternative). We fill the gap they leave: small/non-transformer models with consistent API.

## What we welcome

- 🟢 New `Backend` implementations for architectures we don't support (RWKV, Hyena, RetNet, custom)
- 🟢 Bug fixes and clearer error messages
- 🟢 New `circuit` detectors backed by published papers
- 🟢 Improvements to existing techniques (faster SAE training, better probe initialization, etc.)
- 🟢 Documentation, examples, notebook tutorials
- 🟢 Test coverage improvements

## What is out of scope

- 🔴 Adding production / large-scale features (multi-GPU SAE training at scale, distributed probing) — use `sae_lens` or `transformer_lens` for those
- 🔴 Visualization frontends as part of core (PRs welcome as separate companion packages)
- 🔴 Adding methods that don't have a published paper backing them (we want established techniques)

## Style

- Type hints on public API (not required for internals)
- Docstrings on public functions explaining: what it does, args, return shape
- Tests for new functionality (see `tests/test_unit.py` for examples)
- Apache-2.0 license — by contributing you agree to the same license

## Adding a new Backend

The minimum interface a backend needs:
```python
@Backend.register("my_arch")
class MyArchBackend(Backend):
    def layer_names(self) -> list[str]:
        ...
    def extract(self, inputs, layers=None) -> list[ActivationRecord]:
        ...
    def hidden_dim(self, layer_name: str) -> int:
        ...
```

See `src/polylens/backends.py` (TransformerBackend, MambaBackend) and `src/polylens/kazdov_backend.py` for examples.

## Running tests

```bash
pip install -e .
pip install pytest
pytest tests/test_unit.py -v
```

For full integration tests (download models, ~500MB, ~10 min):
```bash
python tests/test_pythia_end_to_end.py
python tests/test_mamba_integration.py
python tests/run_interpbench_leaderboard.py
```

## Filing issues

- Bug? Include: minimum reproducer, model name, polylens version, transformers version, Python version, full traceback.
- Feature request? Reference the paper or motivation. We prefer techniques with published validation.

## Code of conduct

Be respectful and on-topic. Solo-dev project — responses may take time.
