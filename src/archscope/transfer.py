"""Cross-architecture probe transfer via paired-activation linear alignment.

Setup:
- Two models with different hidden dims (e.g., Pythia 768 vs Kazdov 512)
- Train a linear projection M: kazdov_space (512) → pythia_space (768) using
  paired (text → activation pair) data
- A Pythia probe with weights w_py ∈ R^768 transferred to kazdov-space becomes
  w_kz = M^T · w_py ∈ R^512

This tests: do probe directions from one architecture transfer to another?
"""
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TransferResult:
    """One transfer experiment's metrics."""
    source_arch: str           # e.g., "pythia"
    target_arch: str           # e.g., "kazdov"
    source_layer: str
    target_layer: str
    n_align_pairs: int

    baseline_source_auroc: float  # source probe on source data (no transfer)
    baseline_target_auroc: float  # target probe on target data (in-arch reference)
    transfer_auroc: float         # source probe via M applied to target data
    transfer_drop: float           # baseline_target - transfer (how much we lose)


def learn_alignment(
    src_acts: torch.Tensor,    # (N, d_src)
    tgt_acts: torch.Tensor,    # (N, d_tgt)
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Ridge regression: learn M such that M @ tgt_acts.T ≈ src_acts.T.

    M has shape (d_src, d_tgt). After fit, src_act ≈ M @ tgt_act.

    Ridge term prevents overfitting on small N.
    """
    assert src_acts.shape[0] == tgt_acts.shape[0]
    src_acts = src_acts.float()
    tgt_acts = tgt_acts.float()
    # M = src @ tgt.T @ (tgt @ tgt.T + λI)^-1
    # But the standard formulation: we want X @ M.T = Y where X is target, Y is source
    # So M.T = (X.T X + λI)^-1 X.T Y
    # Then M = ((X.T X + λI)^-1 X.T Y).T
    X = tgt_acts  # (N, d_tgt)
    Y = src_acts  # (N, d_src)
    XtX = X.T @ X
    reg = torch.eye(XtX.shape[0], device=XtX.device) * ridge
    M_T = torch.linalg.solve(XtX + reg, X.T @ Y)
    M = M_T.T  # (d_src, d_tgt)
    return M


def transfer_probe(
    probe_weights_source: torch.Tensor,  # (d_src,) — Pythia probe direction
    probe_bias_source: torch.Tensor,     # scalar
    alignment: torch.Tensor,             # M: (d_src, d_tgt)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport a probe from source arch to target arch via alignment matrix.

    Given Pythia probe f(x_py) = w_py · x_py + b, and alignment x_py ≈ M @ x_kz,
    the kazdov-space probe is w_kz = M.T @ w_py, with same bias.
    """
    w_target = alignment.T @ probe_weights_source
    return w_target, probe_bias_source


def auroc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Quick AUROC."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))
    except ValueError:
        return float("nan")


def evaluate_transfer(
    source_model, target_model,
    source_backend, target_backend,
    source_tokenize, target_tokenize,
    align_texts: list[str],      # texts for learning M (no labels needed)
    train_pos: list[str], train_neg: list[str],
    test_pos: list[str], test_neg: list[str],
    source_layer: str,
    target_layer: str,
    source_arch_name: str = "source",
    target_arch_name: str = "target",
) -> TransferResult:
    """Full transfer experiment.

    1. Get paired source/target activations on `align_texts` → learn alignment M
    2. Train source probe on source activations of train_pos/neg
    3. Train target probe on target activations of train_pos/neg (in-arch baseline)
    4. Transfer source probe via M, apply to target activations of test_pos/neg
    5. Compare AUROCs.
    """
    def _extract_pooled(backend, tokenize, texts: list[str], layer: str) -> torch.Tensor:
        """Extract activations and pool to (N, hidden_dim) by averaging seq dim."""
        with torch.no_grad():
            rec = backend.extract(tokenize(texts), layers=[layer])[0]
            acts = rec.activations.detach()
        return acts.mean(dim=1) if acts.dim() == 3 else acts

    # -- 1. Learn alignment from paired activations on the same texts.
    src_align = _extract_pooled(source_backend, source_tokenize, align_texts, source_layer)
    tgt_align = _extract_pooled(target_backend, target_tokenize, align_texts, target_layer)
    M = learn_alignment(src_align, tgt_align)

    # -- 2. Train source probe (Pythia-style) on source activations
    from . import probes
    src_train_inputs_pos = source_tokenize(train_pos)
    src_train_inputs_neg = source_tokenize(train_neg)
    pf_src = probes.fit_probe(
        source_model, src_train_inputs_pos, src_train_inputs_neg,
        layer_name=source_layer, backend_hint=source_arch_name,
    )

    # Extract source probe weights (linear case)
    src_probe_net = pf_src.probe.net
    if not isinstance(src_probe_net, nn.Linear):
        raise NotImplementedError("transfer only works for linear probes for now")
    w_src = src_probe_net.weight.data.squeeze(0)
    b_src = src_probe_net.bias.data

    # -- 3. Train target probe (in-arch baseline)
    tgt_train_inputs_pos = target_tokenize(train_pos)
    tgt_train_inputs_neg = target_tokenize(train_neg)
    pf_tgt = probes.fit_probe(
        target_model, tgt_train_inputs_pos, tgt_train_inputs_neg,
        layer_name=target_layer, backend_hint=target_arch_name,
    )

    # -- 4. Transfer: project w_src into target space via M
    w_transferred, b_transferred = transfer_probe(w_src, b_src, M)

    # -- 5. Evaluate on the test set.
    with torch.no_grad():
        # Source baseline: source probe on source test data.
        src_test_acts_pos = _extract_pooled(source_backend, source_tokenize, test_pos, source_layer)
        src_test_acts_neg = _extract_pooled(source_backend, source_tokenize, test_neg, source_layer)
        src_scores = torch.cat([
            src_probe_net(src_test_acts_pos).squeeze(-1),
            src_probe_net(src_test_acts_neg).squeeze(-1),
        ])
        src_labels = torch.cat([
            torch.ones(len(src_test_acts_pos)),
            torch.zeros(len(src_test_acts_neg)),
        ])
        baseline_source = auroc_from_scores(src_scores, src_labels)

        # Target baseline: target probe on target test data.
        tgt_test_acts_pos = _extract_pooled(target_backend, target_tokenize, test_pos, target_layer)
        tgt_test_acts_neg = _extract_pooled(target_backend, target_tokenize, test_neg, target_layer)
        tgt_probe_net = pf_tgt.probe.net
        tgt_scores = torch.cat([
            tgt_probe_net(tgt_test_acts_pos).squeeze(-1),
            tgt_probe_net(tgt_test_acts_neg).squeeze(-1),
        ])
        tgt_labels = torch.cat([
            torch.ones(len(tgt_test_acts_pos)),
            torch.zeros(len(tgt_test_acts_neg)),
        ])
        baseline_target = auroc_from_scores(tgt_scores, tgt_labels)

        # Transfer: transferred probe on target activations
        transfer_scores = torch.cat([
            (tgt_test_acts_pos @ w_transferred + b_transferred),
            (tgt_test_acts_neg @ w_transferred + b_transferred),
        ])
        transfer_auroc = auroc_from_scores(transfer_scores, tgt_labels)

    return TransferResult(
        source_arch=source_arch_name,
        target_arch=target_arch_name,
        source_layer=source_layer,
        target_layer=target_layer,
        n_align_pairs=len(align_texts),
        baseline_source_auroc=baseline_source,
        baseline_target_auroc=baseline_target,
        transfer_auroc=transfer_auroc,
        transfer_drop=baseline_target - transfer_auroc,
    )
