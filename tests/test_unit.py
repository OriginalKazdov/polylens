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
    import mechinterp_small
    from mechinterp_small import probes, sae, neurons, attribute, backends, circuits, transfer, bench
    assert mechinterp_small.__version__ == "0.1.0"


def test_sae_dense_synthetic():
    """Dense SAE trains and produces finite recon."""
    from mechinterp_small import sae
    fake = torch.randn(100, 64)
    cfg = sae.SAEConfig(input_dim=64, n_features=128, sae_type="dense")
    trained = sae.fit_sae(fake, cfg, epochs=5)
    assert torch.isfinite(torch.tensor(trained.last_metrics["recon"]))
    assert "n_active" in trained.last_metrics


def test_sae_rank1_synthetic():
    """Rank-1 factored SAE trains and produces finite recon."""
    from mechinterp_small import sae
    fake = torch.randn(100, 64)
    cfg = sae.SAEConfig(input_dim=64, n_features=128, sae_type="rank1")
    trained = sae.fit_sae(fake, cfg, epochs=5)
    assert torch.isfinite(torch.tensor(trained.last_metrics["recon"]))


def test_sae_invalid_type():
    """Unknown SAE type raises."""
    from mechinterp_small import sae
    with pytest.raises(ValueError):
        sae.build_sae(sae.SAEConfig(input_dim=8, n_features=16, sae_type="nonsense"))


def test_backend_registry():
    """Backends are registered and discoverable."""
    from mechinterp_small.backends import Backend
    for name in ("transformer", "mamba", "recurrent"):
        assert name in Backend._registry, f"{name} not registered"


def test_kazdov_backend_registers_when_available():
    """KazdovBackend optional import succeeds."""
    from mechinterp_small.backends import Backend
    # kazdov_backend imports at __init__ and registers — it's optional
    if "kazdov" in Backend._registry:
        # If kazdov repo is importable, backend should be there
        from mechinterp_small.kazdov_backend import KazdovBackend
        assert KazdovBackend is Backend._registry["kazdov"]


def test_alignment_math():
    """Alignment learning produces a matrix that reconstructs source from target."""
    from mechinterp_small.transfer import learn_alignment
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
    from mechinterp_small.probes import ProbeFit, ProbeConfig
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
    from mechinterp_small.circuits import CircuitScore
    s = CircuitScore(name="test", score=1.0, baseline=0.5, relative=2.0, raw={"x": 1})
    assert s.score == 1.0
    assert s.raw["x"] == 1


def test_interpprofile_serializes():
    """InterpProfile is dataclass and JSON-serializable."""
    from dataclasses import asdict
    import json
    from mechinterp_small.bench import InterpProfile
    p = InterpProfile(model_name="test", arch_family="transformer", n_params=1000,
                      n_layers=2, hidden_dim=16)
    j = json.dumps(asdict(p), default=str)
    assert "test" in j


if __name__ == "__main__":
    # Allow `python tests/test_unit.py` for quick local check
    pytest.main([__file__, "-v"])
