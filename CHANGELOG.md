# Changelog

## v0.2.2 — 2026-05-14

**Renamed: `polylens` → `archscope`.**

The PyPI name `polylens` was rejected as "too similar to an existing project".
Rather than namespace it (e.g., `polylens-mech`), we picked a more distinctive
name that also better describes the project: **arch** + **scope** = looking at
architecture internals.

What changed:
- GitHub repo: `OriginalKazdov/polylens` → `OriginalKazdov/archscope`
  (GitHub redirect keeps old links working)
- Python package: `polylens` → `archscope` (both PyPI name and import name)
- CLI: `polylens` → `archscope`
- All citations, docstrings, READMEs, examples updated.

No API behaviour changes. To migrate:

```bash
pip uninstall polylens     # if you installed v0.2.1 from git
pip install archscope
```

Then `import archscope as mi` (was `import polylens as mi`).

## v0.2.1 — 2026-05-14

Polish + reproducibility pass. No new APIs; clearer positioning and one new
script for reproducing results.

### Positioning
- README now opens with **"What archscope is"** — a small-model interpretability
  workbench, explicitly *not* a competitor to `transformer_lens` / `nnsight`.
- Findings rephrased from declarative claims to open questions
  (e.g., "Does X scale with size?" instead of "X scales with size.").
- New **Metrics caveats** section documenting what each metric measures
  (induction = behavioral, SSM variance = descriptive, logit lens = diagnostic).
- `pyproject.toml` description: "Unified mech interp toolkit ..." → "Lightweight
  workbench for cross-architecture mechanistic interpretability experiments on
  small models".
- `MambaBackend` docstring: removed absolute "unique among mech interp libraries"
  claim. Now describes what it exposes + when it's useful.
- `CONTRIBUTING.md` softened — removed jab at `transformer_lens` being
  "transformer-only" (not quite accurate per their docs).

### New affordances
- `scripts/reproduce_mini_zoo.py` — single-command regeneration of the
  README leaderboard. Outputs both `mini_zoo_leaderboard.json` and `.md`.
- `archscope bench MODEL --out report.md` now writes human-readable markdown.
  File extension chooses format: `.md` → markdown, `.json` → JSON.

### Engineering
- Engineering sweep: 39 → 0 ruff issues. Removed 5 unused imports, 3 dead
  local vars, 24 multi-statement single-line blocks.
- Extracted shared `_utils.resolve_unembedding` + `resolve_final_norm`
  helpers; eliminated duplicate `_resolve_module` code between `neurons.py`
  and `attribute.py`.
- Wrapped `TransformerBackend.extract` and `attribute.activation_patch`
  forward passes in `torch.no_grad()` — extraction shouldn't build backward
  graphs.
- `cli.py`: lazy-imported torch/transformers so `archscope info` is fast.

### Renamed (internal)
- The library was renamed from `mechinterp-small` to `archscope` between v0.1
  and v0.2 to better reflect its purpose: a collection of "lenses" across
  model architectures, complementary to `transformer_lens`.

---

## v0.2.0 — 2026-05-14

### New modules
- **`archscope.lens`** — Logit lens + Tuned lens (Belrose et al 2023).
  Project intermediate residual streams to vocab space.
  - `logit_lens(model, tokenizer, prompt, target_token=None)`
  - `TunedLens.fit(model, tokenizer, calibration_texts)` — learned per-layer
    affine translations
- **`archscope.diff`** — Compare base vs fine-tuned models. Returns per-layer
  residual drift, top shifted neurons, and circuit-score deltas.
  - `compare(base, fine_tuned, tokenizer, calibration_texts, backend_hint=...)`
  - Validated: identity diff → 0; perturbing weights → drift concentrates at
    perturbation site.

### Improvements
- Bumped version 0.1.0 → 0.2.0.
- All 9 modules exported via `__all__`.
- `_utils` now also resolves unembedding and final-norm modules.

### Validation
- Mini-zoo study: 7 small models (Pythia-160m/410m, GPT-2, Mamba-130m/370m,
  Qwen2.5-0.5B, kazdov-α) profiled with `bench.benchmark()`. Results saved
  to `_research/mini_zoo_leaderboard.json`. ~10 min total compute on CPU.

### Tests
- 12/12 unit tests pass (`pytest tests/test_unit.py`).
- 6/6 Pythia integration / 6/6 Mamba integration / 6/6 kazdov integration.
- 5/5 SSM-state extraction tests.
- 3/3 lens tests / 2/2 diff tests.

---

## v0.1.0 — 2026-05-14 (initial release, then named `mechinterp-small`)

### Core API
- `backends.Backend` abstraction with auto-detect: `transformer`, `mamba`, `kazdov`, `recurrent`
- `probes.fit_probe` — linear/MLP probes (Drop the Act-style)
- `sae.fit_sae` — Dense + Rank-1 factored SAEs (WriteSAE-style)
- `neurons.find_neurons` — top-k contrastive neuron discovery
- `attribute.activation_patch` + `dim_decompose` — patching + difference-in-means
- `circuits.run_all_circuits` — induction head, copy, concentration detectors
- `transfer.evaluate_transfer` — cross-arch probe transfer via linear alignment
- `bench.benchmark` — InterpBench standardized profile

### Notable features
- **Mamba SSM state extraction** via `layer_N.ssm_state` — exposes the
  recurrent state used by Mamba-style models, not just residual stream.
- Custom `KazdovBackend` for hybrid MoBE-BCN+MHA architectures.
- CLI: `archscope info`, `archscope bench MODEL --arch ARCH`.

### Validated on
- EleutherAI/pythia-160m (HF transformer)
- state-spaces/mamba-130m-hf (SSM, HF format)
- kazdov-α-98m (hybrid MoBE-BCN, custom architecture)
