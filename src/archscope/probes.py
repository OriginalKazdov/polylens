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

    @property
    def direction(self) -> torch.Tensor:
        """1D direction vector in activation space (linear probes only).

        Shape: ``(hidden_dim,)``. This is the projection axis the probe found —
        useful for: applying a probe to externally-transformed activations
        (e.g., after ``archscope.transfer.learn_alignment``), inspecting feature
        geometry, or projecting interventions along the learned direction.

        Raises ``ValueError`` for MLP probes (no single linear direction).
        """
        if self.config.probe_type != "linear":
            raise ValueError(
                f".direction is only defined for linear probes (got "
                f"probe_type={self.config.probe_type!r}). MLP probes don't have a "
                "single direction in activation space."
            )
        return self.probe.net.weight.detach().squeeze(0).clone()

    @property
    def bias(self) -> torch.Tensor:
        """Scalar bias term (linear probes only). Shape: ``()``.

        Together with ``.direction``, lets you score arbitrary activations as
        ``logits = acts @ direction + bias`` without going through the probe
        module — handy for cross-arch transfer experiments.
        """
        if self.config.probe_type != "linear":
            raise ValueError(
                f".bias is only defined for linear probes (got "
                f"probe_type={self.config.probe_type!r})."
            )
        return self.probe.net.bias.detach().squeeze().clone()


def _auroc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC from logits + binary labels.

    Returns 0.5 (chance) when only one class is present in `labels`
    (the typical small-split case) — this is more informative than NaN
    and avoids sklearn's UndefinedMetricWarning leaking to user code.
    """
    import warnings
    from sklearn.metrics import roc_auc_score
    scores = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()
    if len(set(y.tolist())) < 2:
        warnings.warn(
            "Only one class present in this split — AUROC undefined; "
            "returning 0.5 (chance). Increase val_split or dataset size.",
            stacklevel=2,
        )
        return 0.5
    return float(roc_auc_score(y, scores))


# High-level API matching paper

def fit_probe(
    model,
    inputs_pos=None,
    inputs_neg=None,
    layer_name: str = "",
    backend_hint: str | None = None,
    config: ProbeConfig | None = None,
    device: str = "cpu",
    *,
    tokenizer=None,
    pos_texts: list[str] | None = None,
    neg_texts: list[str] | None = None,
    max_length: int = 32,
) -> ProbeFit:
    """End-to-end: extract activations from a model and fit a probe.

    Two calling conventions:

    1. **Pre-tokenized**: pass ``inputs_pos`` and ``inputs_neg`` as already-tokenized
       dicts (with ``input_ids``, optional ``attention_mask``).

    2. **Texts + tokenizer**: pass ``tokenizer=…``, ``pos_texts=[…]``, ``neg_texts=[…]``
       and archscope tokenizes for you. The kazdov backend requires a bool
       attention_mask; we auto-handle that.

    Returns:
        A ``ProbeFit`` with trained probe and ``.metrics`` (train/val AUROC, loss).
    """
    if pos_texts is not None or neg_texts is not None:
        if tokenizer is None:
            raise ValueError("pos_texts/neg_texts require a tokenizer= argument")
        if pos_texts is None or neg_texts is None:
            raise ValueError("provide both pos_texts and neg_texts")
        from .loader import make_tokenize_fn
        tk = make_tokenize_fn(
            tokenizer, max_length=max_length,
            attention_mask_bool=(backend_hint == "kazdov"),
        )
        inputs_pos = tk(pos_texts)
        inputs_neg = tk(neg_texts)
    elif inputs_pos is None or inputs_neg is None:
        raise ValueError("provide either (inputs_pos, inputs_neg) or "
                          "(tokenizer, pos_texts, neg_texts)")

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
