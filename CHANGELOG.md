# Changelog

## v0.1.0 — 2026-05-14 (initial release)

### Core API
- `backends.Backend` abstraction with auto-detect: `transformer`, `mamba`, `kazdov`, `recurrent`
- `probes.fit_probe` — linear/MLP probes (Drop the Act-style)
- `sae.fit_sae` — Dense + Rank-1 factored SAEs (WriteSAE-style)
- `neurons.find_neurons` — top-k contrastive neuron discovery
- `attribute.activation_patch` + `dim_decompose` — patching + difference-in-means
- `circuits.run_all_circuits` — induction head, copy, concentration detectors
- `transfer.evaluate_transfer` — cross-arch probe transfer via linear alignment
- `bench.benchmark` — InterpBench standardized profile

### Unique features
- **Mamba SSM state extraction** via `layer_N.ssm_state` (first open-source impl)
- Custom `KazdovBackend` for hybrid MoBE-BCN+MHA architectures
- CLI: `polylens info`, `polylens bench MODEL --arch ARCH`

### Validated on
- EleutherAI/pythia-160m (HF transformer)
- state-spaces/mamba-130m-hf (SSM, HF format)
- kazdov-α-98m (hybrid MoBE-BCN, custom architecture)

### Tests
- 6/6 end-to-end on Pythia
- 6/6 on kazdov-α
- 6/6 on Mamba (including SSM state)
- 5/5 SSM-specific tests
- Cross-arch probe transfer (Pythia ↔ Kazdov ↔ Mamba, 6 directions)
- 3-arch SAE layer sweep
- 3-arch circuit detection
- InterpBench leaderboard with 3 reference models
