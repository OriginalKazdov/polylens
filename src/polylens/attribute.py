"""Activation patching + DIM decomposition (Multi-Agent Sycophancy, 2605.12991).

Two methods:
- activation_patch: replace activations from one prompt with another at specified
  layer range. Measures how much of the behavioral gap is "restored" by patching.
- dim_decompose: difference-in-means decomposition of attribution per component
  (MLP vs attention).

Use cases:
- Localize behavior to specific layers (e.g., "L14-L18 restores 96.8% of gap")
- Separate attention vs MLP contribution
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .backends import Backend
from ._utils import resolve_layer_module, resolve_subcomponent_module


@dataclass
class PatchResult:
    layer_range: tuple[int, int]
    gap_restored: float       # fraction of behavioral gap closed
    target_metric: str        # what we measured (e.g., "logit_diff")
    baseline_metric: float
    patched_metric: float
    clean_metric: float


def activation_patch(
    model,
    prompt_source: list,      # source of patched-in activations
    prompt_target: list,      # destination where activations are replaced
    layer_indices: list[int], # which layers to patch
    metric_fn,                # fn(model_outputs) → scalar (e.g., logit diff)
    backend_hint: str | None = None,
) -> PatchResult:
    """Run 3 passes: clean source, clean target, patched (target with source's activations).

    Returns the fraction of behavioral gap that patching closes.
    """
    backend = Backend.for_model(model, hint=backend_hint)
    layer_names = [f"layer_{i}.residual" for i in layer_indices]

    # 1. Clean source: get activations
    src_acts = backend.extract(prompt_source, layers=layer_names)

    # 2. Clean target: baseline behavior
    target_clean_out = model(**prompt_target, output_hidden_states=False, return_dict=True)
    clean_metric = metric_fn(target_clean_out)

    # 3. Patched: hook in source activations at requested layers
    hooks = []
    for layer_name, src_rec in zip(layer_names, src_acts):
        idx = int(layer_name.split("_")[1].split(".")[0])
        module = resolve_layer_module(model, f"layer_{idx}.residual")
        if module is None: continue
        src_h = src_rec.activations

        def hook(mod, inp, out, replacement=src_h):
            if isinstance(out, tuple):
                return (replacement,) + out[1:]
            return replacement
        hooks.append(module.register_forward_hook(hook))

    try:
        patched_out = model(**prompt_target, output_hidden_states=False, return_dict=True)
        patched_metric = metric_fn(patched_out)
    finally:
        for h in hooks: h.remove()

    # Source baseline
    src_out = model(**prompt_source, output_hidden_states=False, return_dict=True)
    source_metric = metric_fn(src_out)

    # Gap restored
    gap = source_metric - clean_metric
    if abs(gap) < 1e-9:
        gap_restored = 0.0
    else:
        gap_restored = (patched_metric - clean_metric) / gap

    return PatchResult(
        layer_range=(min(layer_indices), max(layer_indices)),
        gap_restored=float(gap_restored),
        target_metric="custom",
        baseline_metric=float(source_metric),
        patched_metric=float(patched_metric),
        clean_metric=float(clean_metric),
    )


@dataclass
class DIMResult:
    """Difference-in-means attribution per component."""
    components: dict[str, float]    # e.g., {"attention": 0.45, "mlp": 0.02}
    total: float
    layer_range: tuple[int, int]


def dim_decompose(
    model,
    prompt_a: list,
    prompt_b: list,
    layer_indices: list[int],
    metric_fn,
    components: list[str] = ("attention", "mlp"),
    backend_hint: str | None = None,
) -> DIMResult:
    """Decompose behavioral difference into attention vs MLP contributions.

    Strategy: separately patch attention output, then patch MLP output. The
    fraction of gap each closes = its DIM contribution.
    """
    backend = Backend.for_model(model, hint=backend_hint)

    out_a = model(**prompt_a, return_dict=True)
    out_b = model(**prompt_b, return_dict=True)
    metric_a = metric_fn(out_a)
    metric_b = metric_fn(out_b)
    total_gap = metric_a - metric_b

    contributions = {}
    for comp in components:
        # Hook the specific submodule (attention or mlp)
        hooks = []
        src_acts_by_layer = {}
        # Need to capture source activations at this component
        for idx in layer_indices:
            module = resolve_subcomponent_module(model, idx, comp)
            if module is None: continue
            captured = []
            def capture(mod, inp, out, store=captured):
                store.append(out[0] if isinstance(out, tuple) else out)
            hooks.append(module.register_forward_hook(capture))
            src_acts_by_layer[idx] = captured

        model(**prompt_a, return_dict=True)
        for h in hooks: h.remove()

        # Patch: replace component output with captured during prompt_b run
        patch_hooks = []
        for idx in layer_indices:
            if idx not in src_acts_by_layer: continue
            module = resolve_subcomponent_module(model, idx, comp)
            if module is None: continue
            captured_out = src_acts_by_layer[idx][0]
            def patch(mod, inp, out, repl=captured_out):
                if isinstance(out, tuple):
                    return (repl,) + out[1:]
                return repl
            patch_hooks.append(module.register_forward_hook(patch))

        try:
            patched_out = model(**prompt_b, return_dict=True)
            patched_metric = metric_fn(patched_out)
        finally:
            for h in patch_hooks: h.remove()

        contributions[comp] = float((patched_metric - metric_b) / (total_gap + 1e-9))

    return DIMResult(
        components=contributions,
        total=float(total_gap),
        layer_range=(min(layer_indices), max(layer_indices)),
    )


