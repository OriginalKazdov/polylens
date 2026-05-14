"""Probe-Filtered RL — linear/MLP probes over hidden states (Drop the Act, 2605.11467).

Core idea: train a frozen-base probe over residual/hidden states that predicts
some property (faithfulness, refusal, deception). Use probe scores to filter
trajectories during RL training, or as inference-time signals.

Reimplemented from paper (code anonymized for review).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from .backends import Backend


@dataclass
class ProbeConfig:
    """Configuration for a single probe."""
    layer_name: str
    probe_type: str = "linear"   # "linear" or "mlp"
    hidden_dim: int = 64         # only used if mlp
    target: str = "performativity"


class Probe(nn.Module):
    """Linear or MLP probe over a layer's hidden states.

    Trained to predict a scalar property from activations at one layer.
    Frozen base model: probe is the only trainable component.
    """

    def __init__(self, input_dim: int, config: ProbeConfig):
        super().__init__()
        self.config = config
        if config.probe_type == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif config.probe_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown probe_type: {config.probe_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply probe. Accepts (N, hidden_dim) or (N, seq, hidden_dim); returns
        same leading dims with the final hidden_dim collapsed to scalar logits."""
        return self.net(x).squeeze(-1)


class ProbeFit:
    """Fit a probe given (activations, labels) pairs."""

    def __init__(self, config: ProbeConfig, input_dim: int, device: str = "cpu"):
        self.config = config
        self.probe = Probe(input_dim, config).to(device)
        self.device = device

    def train(
        self,
        activations: torch.Tensor,   # (N, hidden_dim) or (N, seq_len, hidden_dim) pooled
        labels: torch.Tensor,         # (N,)
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_split: float = 0.2,
    ) -> dict:
        """Standard supervised fit. Returns train/val AUROC + final loss."""
        if activations.dim() == 3:
            # Pool seq dim by mean (could be configurable)
            activations = activations.mean(dim=1)
        activations = activations.to(self.device)
        labels = labels.float().to(self.device)

        n = len(activations)
        idx = torch.randperm(n)
        n_val = int(n * val_split)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        opt = torch.optim.AdamW(self.probe.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            self.probe.train()
            perm = train_idx[torch.randperm(len(train_idx))]
            for b in range(0, len(perm), batch_size):
                batch = perm[b:b+batch_size]
                logits = self.probe(activations[batch])
                loss = loss_fn(logits, labels[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.probe.eval()
        with torch.no_grad():
            train_logits = self.probe(activations[train_idx])
            val_logits = self.probe(activations[val_idx])
        return {
            "train_auroc": _auroc(train_logits, labels[train_idx]),
            "val_auroc": _auroc(val_logits, labels[val_idx]),
            "train_loss": loss_fn(train_logits, labels[train_idx]).item(),
        }

    def score(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply probe — returns per-token (or per-example if pooled) scores."""
        self.probe.eval()
        with torch.no_grad():
            return torch.sigmoid(self.probe(activations.to(self.device)))


def _auroc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple AUROC from logits + binary labels."""
    from sklearn.metrics import roc_auc_score
    scores = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")   # happens when only one class present


# High-level API matching paper

def fit_probe(
    model,
    inputs_pos: list,    # examples where target=1 (e.g., faithful)
    inputs_neg: list,    # examples where target=0 (e.g., reasoning theater)
    layer_name: str,
    backend_hint: str | None = None,
    config: ProbeConfig | None = None,
    device: str = "cpu",
) -> ProbeFit:
    """End-to-end: extract activations from model, fit probe."""
    backend = Backend.for_model(model, hint=backend_hint)
    config = config or ProbeConfig(layer_name=layer_name)

    # Extract activations for both classes — detach from autograd graph
    # (probe is trained separately; base model is frozen)
    with torch.no_grad():
        acts_pos = backend.extract(inputs_pos, layers=[layer_name])[0].activations.detach()
        acts_neg = backend.extract(inputs_neg, layers=[layer_name])[0].activations.detach()

    # Pool across sequence dim — handles independent padding across batches
    if acts_pos.dim() == 3:
        acts_pos = acts_pos.mean(dim=1)   # (batch, hidden)
    if acts_neg.dim() == 3:
        acts_neg = acts_neg.mean(dim=1)

    labels_pos = torch.ones(len(acts_pos))
    labels_neg = torch.zeros(len(acts_neg))

    all_acts = torch.cat([acts_pos, acts_neg], dim=0)
    all_labels = torch.cat([labels_pos, labels_neg])

    hidden_dim = backend.hidden_dim(layer_name)
    probe_fit = ProbeFit(config, hidden_dim, device=device)
    metrics = probe_fit.train(all_acts, all_labels)
    probe_fit.metrics = metrics
    return probe_fit
