"""Sparse Autoencoders for hidden states (WriteSAE, 2605.12770).

Two SAE variants:
- DenseSAE: standard SAE with L1 sparsity penalty
- Rank1FactoredSAE: WriteSAE's contribution — atoms = v_i w_i^T (rank-1 outer
  product), applied to recurrent cache writes specifically.

Both work on transformer residual streams AND recurrent hidden states.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    input_dim: int
    n_features: int             # dictionary size
    sparsity: float = 1e-3      # L1 coefficient
    sae_type: str = "dense"     # "dense" or "rank1"
    learning_rate: float = 1e-3
    batch_size: int = 64


class DenseSAE(nn.Module):
    """Standard SAE: encoder + decoder, L1 sparsity on hidden code."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.n_features)
        self.decoder = nn.Linear(config.n_features, config.input_dim, bias=False)
        # Initialize decoder weights to encoder.T transpose (helpful for SAEs)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor):
        x_hat, z = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        l1 = z.abs().mean() * self.config.sparsity
        return recon + l1, {"recon": recon.item(), "l1": l1.item(), "n_active": (z > 0).float().mean().item()}


class Rank1FactoredSAE(nn.Module):
    """WriteSAE rank-1 factored atoms: each feature i = v_i w_i^T outer product.

    Designed for recurrent CACHE WRITES specifically — atoms substitute for
    native write contributions at matched Frobenius norm.

    Reference: Eq. 3-factor closed form Δℓ ≈ G·⟨w_i,q_t⟩·⟨v_i,W_u[tok]⟩
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        assert config.sae_type == "rank1"
        self.config = config
        # Atoms parameterized as v (output) and w (input) vectors
        self.v = nn.Parameter(torch.randn(config.n_features, config.input_dim) * 0.02)
        self.w = nn.Parameter(torch.randn(config.n_features, config.input_dim) * 0.02)

    def fire(self, x: torch.Tensor) -> torch.Tensor:
        """Firing strengths per atom: <w_i, x>, kept positive."""
        # x: (..., input_dim), w: (n_features, input_dim)
        return F.relu(x @ self.w.T)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        fires = self.fire(x)
        # Output: sum_i fires_i * v_i
        return fires @ self.v

    def forward(self, x: torch.Tensor):
        x_hat = self.reconstruct(x)
        z = self.fire(x)
        return x_hat, z

    def loss(self, x: torch.Tensor):
        x_hat, z = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        l1 = z.abs().mean() * self.config.sparsity
        return recon + l1, {"recon": recon.item(), "l1": l1.item(), "n_active": (z > 0).float().mean().item()}


def build_sae(config: SAEConfig):
    if config.sae_type == "dense":
        return DenseSAE(config)
    if config.sae_type == "rank1":
        return Rank1FactoredSAE(config)
    raise ValueError(f"Unknown sae_type: {config.sae_type}")


def fit_sae(
    activations: torch.Tensor,   # (N, input_dim) flattened
    config: SAEConfig,
    epochs: int = 100,
    device: str = "cpu",
) -> nn.Module:
    """Train SAE on flattened activations."""
    sae = build_sae(config).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=config.learning_rate)
    activations = activations.to(device)

    n = len(activations)
    last_metrics = {}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, config.batch_size):
            batch = activations[perm[b:b+config.batch_size]]
            loss, metrics = sae.loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            last_metrics = metrics
    sae.last_metrics = last_metrics
    return sae
