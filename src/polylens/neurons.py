"""Targeted Neuron Modulation via Contrastive Pair Search (2605.12290).

5-step algorithm reimplemented from paper:
1. Run harmful + benign prompts through model
2. Record MLP activations at final token position
3. Compute per-neuron mean activation difference
4. Select top k (default 0.1%) neurons by |Δ|
5. At inference: multiply selected neuron activations by scalar m
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .backends import Backend
from ._utils import resolve_layer_module


@dataclass
class NeuronEditConfig:
    top_frac: float = 0.001         # top 0.1% by default
    layer_filter: str | None = None # e.g., "mlp" to restrict to MLP neurons
    mode: str = "scalar"            # "scalar" (multiply by m) or "ablate" (m=0)


@dataclass
class NeuronEdit:
    """Result of contrastive search — which neurons to edit + how."""
    layer_to_indices: dict[str, torch.Tensor]   # layer_name → neuron indices
    layer_to_deltas: dict[str, torch.Tensor]    # layer_name → mean diff values
    config: NeuronEditConfig
    multiplier: float = 0.0   # 0 = ablate; >1 = amplify; <1 = dampen

    def apply_hook(self, model, backend: Backend | None = None):
        """Register forward hooks that modulate the selected neurons.

        Returns a context manager that auto-removes hooks on exit.
        """
        hooks = []
        for layer_name, indices in self.layer_to_indices.items():
            module = resolve_layer_module(model, layer_name)
            if module is None:
                continue

            indices_local = indices

            def hook(module, input, output, idxs=indices_local, m=self.multiplier):
                # Return MODIFIED output (don't rely on in-place — some HF layers
                # produce fresh tensors that won't propagate in-place changes).
                if isinstance(output, tuple):
                    h = output[0].clone()
                    if h.dim() >= 2:
                        h[..., idxs] = h[..., idxs] * m
                    return (h,) + output[1:]
                h = output.clone()
                if h.dim() >= 2:
                    h[..., idxs] = h[..., idxs] * m
                return h
            hooks.append(module.register_forward_hook(hook))

        return _HookContext(hooks)


class _HookContext:
    """Auto-removes forward hooks when used as a `with` block."""

    def __init__(self, hooks):
        self.hooks = hooks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()


def find_neurons(
    model,
    inputs_harmful: list,
    inputs_benign: list,
    config: NeuronEditConfig | None = None,
    backend_hint: str | None = None,
) -> NeuronEdit:
    """Algorithm 1 of Targeted Neuron Modulation paper.

    Returns a NeuronEdit that can be applied as a hook during inference.
    """
    config = config or NeuronEditConfig()
    backend = Backend.for_model(model, hint=backend_hint)

    # Get all layers (will filter to MLP later if requested)
    all_layers = backend.layer_names()

    # Forward both classes, collect final-token activations
    harm_acts = backend.extract(inputs_harmful, layers=all_layers)
    ben_acts = backend.extract(inputs_benign, layers=all_layers)

    layer_to_indices = {}
    layer_to_deltas = {}
    for h_rec, b_rec in zip(harm_acts, ben_acts):
        # Final token: (batch, -1, hidden_dim) -> (batch, hidden_dim)
        h_final = h_rec.activations[:, -1, :]
        b_final = b_rec.activations[:, -1, :]
        # Per-neuron mean diff
        delta = h_final.mean(dim=0) - b_final.mean(dim=0)
        # Top k by absolute value
        k = max(1, int(config.top_frac * len(delta)))
        topk = torch.topk(delta.abs(), k=k)
        layer_to_indices[h_rec.layer_name] = topk.indices
        layer_to_deltas[h_rec.layer_name] = delta[topk.indices]

    return NeuronEdit(
        layer_to_indices=layer_to_indices,
        layer_to_deltas=layer_to_deltas,
        config=config,
        multiplier=0.0,
    )


