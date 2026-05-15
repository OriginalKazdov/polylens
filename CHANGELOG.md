# Changelog

## v0.2.6 — 2026-05-14

Second-round audit shook out 4 more issues. The v0.2.5 changelog also claimed
a vocab-range fix in `circuits.induction_head_score` that was documented but
never actually applied — that's also fixed here.

### Fixed (HIGH)
- **`circuits.induction_head_score` adaptive vocab window** — v0.2.5 changelog
  promised this and shipped without it. Window is now
  `[min(100, vocab//4), min(vocab, 40000))` and raises a clear `ValueError`
  if it can't fit `2 * n_pairs` distinct tokens.
- **`attribute.activation_patch` capture clone** — same class of silent
  data-corruption bug v0.2.5 fixed in `dim_decompose`. The patch hook stored
  a reference to the source-extraction tensor instead of a clone; for some HF
  paths (KV-cache reuse, gradient checkpointing) the second forward pass
  could mutate it.
- **`lens.TunedLens.fit` trained on PAD positions** when calibration_texts
  had varying length. Padding was enabled but `[:, -1, :]` selects the last
  pad token for short rows. Now uses `attention_mask` to find each row's
  real last-token position. Also auto-sets `pad_token = eos_token` if the
  tokenizer lacks one (e.g. GPT-2 family).

### Fixed (UX)
- **`attribute.dim_decompose` raises on non-transformer architectures**
  instead of silently returning a `DIMResult` with empty `contributions`.
  Matches the strictness of `Backend.for_model`.

### Docs
- README's first code snippet now uses the documented `load_model` one-call
  helper instead of duplicating the HF boilerplate. The helper was added in
  v0.2.3 but the README never showcased it.
- New **method × backend support matrix** in the README — answers "does
  `dim_decompose` work on Mamba?" without grepping the source. Honest about
  what's transformer-only and where lens degrades.
- Cleaned a stale comment in `TransformerBackend.layer_names` (referenced
  `model.model.layers[i]` even though extraction goes through HF's
  `output_hidden_states=True` — there's no direct attribute walk).

## v0.2.5 — 2026-05-14

Independent agent audit shook out 8 correctness/promise gaps. All fixed.

### Fixed (HIGH)
- **`attribute.dim_decompose` was silently wrong.** The capture hook stored a
  *reference* to the component output tensor; the next forward pass reused the
  same module buffers and overwrote it. DIM attributions could degrade to zero
  with no error. Now `detach().clone()` at capture time.
- **`attribute.activation_patch` shape mismatch could install a hook that
  crashed deep inside the model.** Now raises a clear `ValueError` upfront if
  `prompt_source.input_ids` and `prompt_target.input_ids` have different
  shapes.
- **`Backend.for_model` autodetect silently fell through to `RecurrentBackend`
  for any `config.model_type` it didn't recognize**, including several models
  the README claimed worked. The autodetect table is now explicit (19 model
  types — Llama/Mistral/Qwen2/Qwen3/Pythia/GPT-Neo/GPT-J/Falcon/MPT/Bloom/OPT/
  Phi/Phi-3/Gemma/Gemma-2/StarCoder2/Mamba/Mamba-2 + GPT-2) and raises
  `ValueError` for anything else. Pass `hint="..."` to use a backend for an
  unrecognized model_type.
- **`archscope bench --arch mamba` always returned `ssm_state_variance_ratio
  = NaN`** because the CLI didn't pass `ssm_layer`. Now defaults to mid-depth.

### Fixed (MED)
- **`circuits.induction_head_score` hardcoded a `[100, 40000)` vocab window**
  and broke on small-vocab models. Adapts to `vocab_size` with a clear error
  if there isn't enough headroom for `n_pairs` distinct tokens.
- **`circuits.copy_score` assumed BPE leading-space tokens (`" word"`).**
  Failed silently on SentencePiece tokenizers (Llama-3/Qwen). Falls back to
  the bare word and skips pathologically empty encodings.
- **`diff.compare` crashed on tokenizers without a `pad_token`** (GPT-2 family
  ships without one). Mirrors `loader.load_model` by setting
  `pad_token = eos_token` when missing.
- **`neurons.NeuronEditConfig.layer_filter` was documented but never
  applied.** Now used as a substring filter on `backend.layer_names()`, with
  a clear error if the filter matches no layers.

### Docs
- README backend table now lists the exact `config.model_type` strings that
  auto-detect, and is explicit that anything else needs `hint="..."`.

## v0.2.4 — 2026-05-14

Engineering hygiene release. No new API; existing API gets honest about
what works for whom.

### Fixed
- **`kazdov_backend` no longer ships dead code.** Previously the
  `load_kazdov_checkpoint()` function was importable from the PyPI
  package but only worked on the maintainer's machine (it hardcoded
  `~/code/OriginalKazdov/kazdov/`). The function has been moved to
  `scripts/_kazdov_loader.py`, which is NOT shipped to PyPI.

  The `KazdovBackend` class itself stays — it's actually generic: it
  works on any PyTorch model exposing `model.blocks` (a `ModuleList`)
  and `model.d_model` (or `hidden_size`). The docstring now documents
  this explicitly. So users with custom architectures CAN register a
  model via `Backend.for_model(model, hint="kazdov")` — they just need
  to load it themselves.

- **`KazdovBackend.hidden_dim()`** now also handles `model.hidden_size`
  (not just `model.d_model`).

### Cleaned up
- All test files (`tests/*.py`) used to hardcode
  `/Users/kazdov/code/OriginalKazdov/archscope/src` in `sys.path.insert`.
  Replaced with `Path(__file__).parent.parent / "src"` so anyone
  who clones the repo can run the tests.
- Kazdov-checkpoint path is now overridable via the `KAZDOV_CHECKPOINT`
  environment variable; the maintainer's default is kept as fallback.

### Performance verified on Pythia-160m (CPU)
- `load_model`:                  3.5 s
- `fit_probe` (sentiment, n=16): 0.16 s
- `logit_lens` (12 layers):      0.07 s
- `TunedLens.fit` (10 epochs):   2.25 s
- Dense SAE (50 epochs, n=144):  0.11 s
- All 3 circuits:                4.69 s
- Full `bench.benchmark`:        6.39 s

## v0.2.3 — 2026-05-14

Engineering pass focused on developer experience. No new methods; existing
APIs got smoother + clearer error paths.

### New
- **`archscope.load_model(name, arch=...)`** — one-call HuggingFace
  model + tokenizer + backend loader. Eliminates ~5 lines of boilerplate
  per example.
- **`archscope.make_tokenize_fn(tokenizer, ...)`** — public helper for
  building a tokenize function (handles kazdov's bool attention_mask).
- **`probes.fit_probe` now accepts `tokenizer + pos_texts + neg_texts`**
  in addition to the pre-tokenized form. Pick whichever fits your code.
- **`archscope --version` / `-V`** — standard CLI version flag.
- **`py.typed` marker** — IDEs now pick up archscope's type hints
  natively.

### Better errors
- `Backend.extract([..., "layer_999.residual"])` now raises a clear
  `ValueError` listing valid layer names. Was: cryptic IndexError.
- `_auroc()` returns `0.5` with a single `UserWarning` when only one
  class is present in a probe split, instead of `NaN` + sklearn's
  `UndefinedMetricWarning`.

### Tests
- 15/15 unit tests pass (was 12). New tests: `test_loader_exports`,
  `test_layer_name_validation_clear_error`, `test_auroc_returns_chance_on_single_class`.

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
