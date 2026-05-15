"""Model Diff — compare base model vs fine-tuned version, find what changed.

This is the practical question every fine-tuner asks:
  "I fine-tuned X on my data. What did it change in the model's brain?"

Given (base_model, fine_tuned_model) with the same architecture, archscope.diff
returns a structured ModelDiff with:
- Per-layer activation drift (how much each layer's residual stream moved)
- Top-K shifted neurons per layer
- Circuit score deltas (induction, copy, etc.)
- Optional: probe direction drift (do trained probes still work?)

Requirements:
- base and fine_tuned must share architecture + tokenizer
- A small calibration_texts list (16-100 short texts) to measure drift on

Usage:
    from archscope.diff import compare
    result = compare(base, finetuned, tokenizer, calibration_texts, backend_hint="transformer")
    print(result.to_markdown())
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

from .backends import Backend
from . import circuits


@dataclass
class LayerDrift:
    """How much one layer's residual stream changed under fine-tuning."""
    layer: int
    layer_name: str
    mean_l2_delta: float           # mean L2 norm of (a_ft - a_base) per token
    relative_drift: float           # mean_l2_delta / mean ||a_base||
    cosine_similarity: float        # mean cos(a_base, a_ft) — closer to 1 = less change
    top_shifted_neurons: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class CircuitDelta:
    """How a single circuit score changed."""
    name: str
    base_score: float
    fine_tuned_score: float
    delta: float                   # ft - base
    relative_change: float          # delta / |base| (capped at large values)


@dataclass
class ModelDiff:
    """Full diff between base and fine-tuned model."""
    arch_family: str
    n_layers: int
    n_calibration_texts: int
    layer_drift: list[LayerDrift] = field(default_factory=list)
    circuit_deltas: list[CircuitDelta] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def top_changed_layers(self, k: int = 3) -> list[LayerDrift]:
        """Return the k layers with highest relative drift."""
        return sorted(self.layer_drift, key=lambda d: -d.relative_drift)[:k]

    def to_markdown(self) -> str:
        lines = [f"# Model Diff — {self.arch_family} ({self.n_layers} layers)\n"]
        lines.append(f"Calibration texts: {self.n_calibration_texts}\n")

        # Layer drift table
        lines.append("## Per-layer residual drift\n")
        lines.append("| Layer | mean ‖Δa‖ | relative | cosine sim |")
        lines.append("|------:|---------:|---------:|----------:|")
        for d in self.layer_drift:
            lines.append(
                f"| {d.layer:>5} | {d.mean_l2_delta:>8.3f} | {d.relative_drift:>7.3%} | {d.cosine_similarity:>9.3f} |"
            )

        # Top changed layers
        lines.append("\n## Top-3 most-changed layers\n")
        top = self.top_changed_layers(3)
        for d in top:
            lines.append(f"- **layer {d.layer}**: relative drift {d.relative_drift:.1%}, cosine {d.cosine_similarity:.3f}")
            if d.top_shifted_neurons:
                idxs = ", ".join(f"#{i}({delta:+.2f})" for i, delta in d.top_shifted_neurons[:5])
                lines.append(f"  - top-5 neurons shifted: {idxs}")

        # Circuit deltas
        if self.circuit_deltas:
            lines.append("\n## Circuit deltas (induction, copy, concentration)\n")
            lines.append("| Circuit | base | fine-tuned | Δ | Δ% |")
            lines.append("|---------|-----:|-----------:|--:|---:|")
            for c in self.circuit_deltas:
                pct = f"{c.relative_change:+.1%}" if abs(c.base_score) > 1e-9 else "—"
                lines.append(
                    f"| {c.name} | {c.base_score:.3f} | {c.fine_tuned_score:.3f} | {c.delta:+.3f} | {pct} |"
                )

        if self.notes:
            lines.append("\n## Notes\n")
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines)


def _activation_drift(
    base_acts: torch.Tensor,
    ft_acts: torch.Tensor,
    top_k_neurons: int = 10,
) -> tuple[float, float, float, list[tuple[int, float]]]:
    """Compute drift stats between two activation tensors of shape (B, T, H) or (B, H).

    Returns (mean_l2_delta, relative_drift, cosine_sim, top_k_shifted_neurons).
    """
    if base_acts.shape != ft_acts.shape:
        raise ValueError(f"Shape mismatch: {tuple(base_acts.shape)} vs {tuple(ft_acts.shape)}")
    # Flatten leading dims; keep hidden dim
    H = base_acts.shape[-1]
    a = base_acts.reshape(-1, H)
    b = ft_acts.reshape(-1, H)
    diff = (b - a).float()
    l2_per_token = diff.norm(dim=-1)
    mean_l2 = float(l2_per_token.mean().item())
    base_l2 = float(a.float().norm(dim=-1).mean().item()) + 1e-9
    relative = mean_l2 / base_l2
    # Cosine sim (per-token, then averaged)
    cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    cosine_sim = float(cos.mean().item())
    # Per-neuron drift: mean absolute delta per channel
    per_neuron_delta = diff.abs().mean(dim=0)   # (H,)
    topk = torch.topk(per_neuron_delta, k=min(top_k_neurons, H))
    top_shifted = [(int(i), float(v)) for i, v in zip(topk.indices.tolist(), topk.values.tolist())]
    return mean_l2, relative, cosine_sim, top_shifted


def compare(
    base_model,
    fine_tuned_model,
    tokenizer,
    calibration_texts: list[str],
    backend_hint: str | None = None,
    run_circuits: bool = True,
    max_length: int = 32,
    device: str = "cpu",
) -> ModelDiff:
    """Compare base vs fine-tuned model on calibration texts.

    Both models must share the same architecture and tokenizer.
    """
    base_backend = Backend.for_model(base_model, hint=backend_hint)
    ft_backend = Backend.for_model(fine_tuned_model, hint=backend_hint)

    layer_names = [n for n in base_backend.layer_names() if ".residual" in n]
    ft_layer_names = [n for n in ft_backend.layer_names() if ".residual" in n]
    if layer_names != ft_layer_names:
        raise ValueError("base and fine_tuned have different layer structure — "
                         "they must share architecture")

    # Ensure tokenizer has a pad token (GPT-2 family ships without one).
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize calibration
    enc = tokenizer(calibration_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length)
    inputs = {"input_ids": enc["input_ids"].to(device)}
    if "attention_mask" in enc:
        inputs["attention_mask"] = enc["attention_mask"].to(device)

    # Need attention_mask as bool for kazdov backend
    if backend_hint == "kazdov" and "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].bool()

    with torch.no_grad():
        base_records = base_backend.extract(inputs, layers=layer_names)
        ft_records = ft_backend.extract(inputs, layers=layer_names)

    diff = ModelDiff(
        arch_family=backend_hint or "auto",
        n_layers=len(layer_names),
        n_calibration_texts=len(calibration_texts),
    )

    for base_rec, ft_rec in zip(base_records, ft_records):
        mean_l2, rel, cos_sim, top_neurons = _activation_drift(
            base_rec.activations, ft_rec.activations,
        )
        idx = int(base_rec.layer_name.split("_")[1].split(".")[0])
        diff.layer_drift.append(LayerDrift(
            layer=idx, layer_name=base_rec.layer_name,
            mean_l2_delta=mean_l2,
            relative_drift=rel,
            cosine_similarity=cos_sim,
            top_shifted_neurons=top_neurons,
        ))

    # Circuit deltas
    if run_circuits:
        try:
            base_circs = circuits.run_all_circuits(base_model, tokenizer=tokenizer, device=device)
            ft_circs = circuits.run_all_circuits(fine_tuned_model, tokenizer=tokenizer, device=device)
            for name in base_circs:
                if name not in ft_circs:
                    continue
                bs = base_circs[name].score
                fs = ft_circs[name].score
                delta = fs - bs
                rel = delta / (abs(bs) + 1e-9)
                diff.circuit_deltas.append(CircuitDelta(
                    name=name,
                    base_score=bs, fine_tuned_score=fs,
                    delta=delta, relative_change=rel,
                ))
        except Exception as e:
            diff.notes.append(f"circuit comparison error: {str(e)[:80]}")

    return diff
