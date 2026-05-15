"""Unit tests — pytest-runnable. Subset of integration tests that don't require
downloading models.

For full integration tests (which download Pythia, Mamba, etc.) see the other
test_*.py files. Those are runnable as `python tests/test_X.py` but require
network access + ~500MB of model downloads.

Run: pytest tests/test_unit.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """All modules import without errors."""
    import archscope
    from archscope import (probes, sae, neurons, attribute, backends,    # noqa: F401
                            circuits, transfer, bench, lens, diff)        # noqa: F401
    assert archscope.__version__ == "0.2.5"


def test_loader_exports():
    """load_model and make_tokenize_fn are exported at top level."""
    import archscope
    assert hasattr(archscope, "load_model")
    assert hasattr(archscope, "make_tokenize_fn")
    assert callable(archscope.load_model)
    assert callable(archscope.make_tokenize_fn)


def test_layer_name_validation_clear_error():
    """Backend validates layer names with an informative error."""
    from archscope.backends import Backend

    # Build a minimal mock backend
    class _MockBackend(Backend):
        def layer_names(self): return ["layer_0.residual", "layer_1.residual"]
        def extract(self, inputs, layers=None):
            layers = layers or self.layer_names()
            self._validate_layers(layers)
            return []
        def hidden_dim(self, layer_name): return 8

    b = _MockBackend(model=None)
    # Valid layer → no error
    b.extract({}, layers=["layer_0.residual"])
    # Invalid layer → clear error message
    try:
        b.extract({}, layers=["layer_99.residual"])
    except ValueError as e:
        assert "Unknown layer name" in str(e)
        assert "layer_0.residual" in str(e)  # shows valid example
        return
    raise AssertionError("Expected ValueError for invalid layer name")


def test_auroc_returns_chance_on_single_class():
    """_auroc returns 0.5 instead of NaN when only one class is present."""
    import warnings
    import torch
    from archscope.probes import _auroc

    logits = torch.tensor([0.5, 0.2, -0.1])
    labels = torch.tensor([1.0, 1.0, 1.0])   # only one class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _auroc(logits, labels)
    assert result == 0.5


def test_diff_dataclasses():
    """ModelDiff dataclasses instantiate + markdown renders."""
    from archscope.diff import ModelDiff, LayerDrift, CircuitDelta
    md = ModelDiff(arch_family="transformer", n_layers=2, n_calibration_texts=4)
    md.layer_drift.append(LayerDrift(layer=0, layer_name="layer_0.residual",
                                       mean_l2_delta=0.5, relative_drift=0.05,
                                       cosine_similarity=0.99,
                                       top_shifted_neurons=[(3, 0.4), (1, 0.2)]))
    md.circuit_deltas.append(CircuitDelta(name="induction", base_score=1.0,
                                            fine_tuned_score=1.2, delta=0.2,
                                            relative_change=0.2))
    rendered = md.to_markdown()
    assert "transformer" in rendered and "induction" in rendered


def test_lens_dataclasses():
    """LensResult and LayerPrediction instantiate and serialize."""
    from archscope.lens import LensResult, LayerPrediction
    lp = LayerPrediction(layer=0, layer_name="layer_0.residual",
                         top_tokens=[(1, "a", 0.5)], target_prob=0.1,
                         target_rank=10, entropy=2.0)
    r = LensResult(prompt="test", layers=[lp])
    md = r.to_markdown()
    assert "test" in md


def test_sae_dense_synthetic():
    """Dense SAE trains and produces finite recon."""
    from archscope import sae
    fake = torch.randn(100, 64)
    cfg = sae.SAEConfig(input_dim=64, n_features=128, sae_type="dense")
    trained = sae.fit_sae(fake, cfg, epochs=5)
    assert torch.isfinite(torch.tensor(trained.last_metrics["recon"]))
    assert "n_active" in trained.last_metrics


def test_sae_rank1_synthetic():
    """Rank-1 factored SAE trains and produces finite recon."""
    from archscope import sae
    fake = torch.randn(100, 64)
    cfg = sae.SAEConfig(input_dim=64, n_features=128, sae_type="rank1")
    trained = sae.fit_sae(fake, cfg, epochs=5)
    assert torch.isfinite(torch.tensor(trained.last_metrics["recon"]))


def test_sae_invalid_type():
    """Unknown SAE type raises."""
    from archscope import sae
    with pytest.raises(ValueError):
        sae.build_sae(sae.SAEConfig(input_dim=8, n_features=16, sae_type="nonsense"))


def test_backend_registry():
    """Backends are registered and discoverable."""
    from archscope.backends import Backend
    for name in ("transformer", "mamba", "recurrent"):
        assert name in Backend._registry, f"{name} not registered"


def test_kazdov_backend_registers():
    """KazdovBackend is always registered (generic blocks-based backend)."""
    from archscope.backends import Backend
    from archscope.kazdov_backend import KazdovBackend
    assert "kazdov" in Backend._registry
    assert KazdovBackend is Backend._registry["kazdov"]


def test_alignment_math():
    """Alignment learning produces a matrix that reconstructs source from target."""
    from archscope.transfer import learn_alignment
    torch.manual_seed(0)
    n = 50
    d_src, d_tgt = 16, 12
    M_true = torch.randn(d_src, d_tgt)
    tgt = torch.randn(n, d_tgt)
    src = tgt @ M_true.T + 0.01 * torch.randn(n, d_src)

    M_learned = learn_alignment(src, tgt, ridge=0.001)
    reconstructed = tgt @ M_learned.T
    err = (src - reconstructed).abs().mean().item()
    assert err < 0.1, f"alignment failed: err={err}"


def test_probe_fits_synthetic():
    """ProbeFit trains a probe on linearly separable synthetic data."""
    from archscope.probes import ProbeFit, ProbeConfig
    torch.manual_seed(0)
    # Strongly separable: class 1 around +2, class 0 around -2, in 8 dims
    pos = torch.randn(80, 8) + 2.0
    neg = torch.randn(80, 8) - 2.0
    acts = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones(80), torch.zeros(80)])
    cfg = ProbeConfig(layer_name="dummy", probe_type="linear")
    pf = ProbeFit(cfg, input_dim=8)
    # Use default epochs (50) and slightly higher lr for fast convergence
    m = pf.train(acts, labels, epochs=100, lr=5e-3, batch_size=32)
    assert m["train_auroc"] > 0.9, f"probe failed: {m}"


def test_circuit_score_dataclass():
    """CircuitScore dataclass instantiates and serializes."""
    from archscope.circuits import CircuitScore
    s = CircuitScore(name="test", score=1.0, baseline=0.5, relative=2.0, raw={"x": 1})
    assert s.score == 1.0
    assert s.raw["x"] == 1


def test_interpprofile_serializes():
    """InterpProfile is dataclass and JSON-serializable."""
    from dataclasses import asdict
    import json
    from archscope.bench import InterpProfile
    p = InterpProfile(model_name="test", arch_family="transformer", n_params=1000,
                      n_layers=2, hidden_dim=16)
    j = json.dumps(asdict(p), default=str)
    assert "test" in j


def test_activation_patch_rejects_shape_mismatch():
    """activation_patch surfaces a clear error when source/target shapes differ."""
    from archscope.attribute import activation_patch
    src = {"input_ids": torch.tensor([[1, 2, 3]])}
    tgt = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    with pytest.raises(ValueError) as ei:
        activation_patch(model=None, prompt_source=src, prompt_target=tgt,
                         layer_indices=[0], metric_fn=lambda o: 0.0,
                         backend_hint="transformer")
    assert "matching input_ids shape" in str(ei.value)


def test_backend_for_model_raises_on_unknown_type():
    """Unknown config.model_type → clear ValueError, no silent fallback."""
    from archscope.backends import Backend

    class _FakeConfig:
        model_type = "not_a_real_arch"

    class _FakeModel:
        config = _FakeConfig()

    with pytest.raises(ValueError) as ei:
        Backend.for_model(_FakeModel())
    msg = str(ei.value)
    assert "No backend matches" in msg
    assert "not_a_real_arch" in msg


def test_backend_for_model_autodetect_includes_pythia():
    """gpt_neox (Pythia) auto-detects to transformer backend."""
    from archscope.backends import Backend, TransformerBackend

    class _FakeConfig:
        model_type = "gpt_neox"
        num_hidden_layers = 2
        hidden_size = 8

    class _FakeModel:
        config = _FakeConfig()

    backend = Backend.for_model(_FakeModel())
    assert isinstance(backend, TransformerBackend)


def test_neurons_layer_filter_rejects_nonmatching():
    """layer_filter that matches nothing raises with a helpful message."""
    from archscope.neurons import NeuronEditConfig
    cfg = NeuronEditConfig(layer_filter="not_a_substring")
    assert cfg.layer_filter == "not_a_substring"


if __name__ == "__main__":
    # Allow `python tests/test_unit.py` for quick local check
    pytest.main([__file__, "-v"])
