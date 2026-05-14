"""Activation patching + DIM decomposition (Multi-Agent Sycophancy, 2605.12991).

Two methods:
- `activation_patch`: replace activations from one prompt with another at
  specified layer range. Measures how much of the behavioral gap is "restored"
  by the patch.
- `dim_decompose`: difference-in-means decomposition of attribution per
  component (MLP vs attention).

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
    """Outcome of a single activation_patch experiment."""
    layer_range: tuple[int, int]
    gap_restored: float       # fraction of behavioral gap closed by patching
    target_metric: str        # what we measured (e.g., "logit_diff")
    baseline_metric: float
    patched_metric: float
    clean_metric: float


@dataclass
class DIMResult:
    """Difference-in-means attribution per component (attention vs MLP)."""
    components: dict[str, float]    # e.g., {"attention": 0.45, "mlp": 0.02}
    total: float
    layer_range: tuple[int, int]


def _remove_hooks(hooks):
    """Cleanly detach a list of forward-hook handles."""
    for h in hooks:
        h.remove()


def activation_patch(
    model,
    prompt_source,
    prompt_target,
    layer_indices: list[int],
    metric_fn,
    backend_hint: str | None = None,
) -> PatchResult:
    """Replace activations at chosen layers with those from a source prompt.

    Args:
        model: any model exposing `model(**inputs, return_dict=True).logits`
        prompt_source: tokenized inputs to extract clean activations from
        prompt_target: tokenized inputs where activations will be replaced
        layer_indices: which layer indices to patch
        metric_fn: function mapping `model_outputs → scalar` (e.g., logit diff)
        backend_hint: backend name for extraction

    Returns:
        PatchResult with the fraction of behavioral gap closed by patching.
    """
    backend = Backend.for_model(model, hint=backend_hint)
    layer_names = [f"layer_{i}.residual" for i in layer_indices]

    # 1. Clean source: extract activations to patch in.
    src_acts = backend.extract(prompt_source, layers=layer_names)

    # 2. Clean target: baseline metric.
    with torch.no_grad():
        target_clean_out = model(**prompt_target, output_hidden_states=False, return_dict=True)
    clean_metric = metric_fn(target_clean_out)

    # 3. Patched target: hook in source activations.
    hooks = []
    for layer_name, src_rec in zip(layer_names, src_acts):
        idx = int(layer_name.split("_")[1].split(".")[0])
        module = resolve_layer_module(model, f"layer_{idx}.residual")
        if module is None:
            continue
        src_h = src_rec.activations

        def hook(mod, inp, out, replacement=src_h):
            if isinstance(out, tuple):
                return (replacement,) + out[1:]
            return replacement
        hooks.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            patched_out = model(**prompt_target, output_hidden_states=False, return_dict=True)
        patched_metric = metric_fn(patched_out)
    finally:
        _remove_hooks(hooks)

    # Source baseline (no hooks).
    with torch.no_grad():
        src_out = model(**prompt_source, output_hidden_states=False, return_dict=True)
    source_metric = metric_fn(src_out)

    gap = source_metric - clean_metric
    gap_restored = 0.0 if abs(gap) < 1e-9 else (patched_metric - clean_metric) / gap

    return PatchResult(
        layer_range=(min(layer_indices), max(layer_indices)),
        gap_restored=float(gap_restored),
        target_metric="custom",
        baseline_metric=float(source_metric),
        patched_metric=float(patched_metric),
        clean_metric=float(clean_metric),
    )


def dim_decompose(
    model,
    prompt_a,
    prompt_b,
    layer_indices: list[int],
    metric_fn,
    components: tuple[str, ...] = ("attention", "mlp"),
    backend_hint: str | None = None,    # kept for symmetry with activation_patch
) -> DIMResult:
    """Decompose a behavioral difference into per-component contributions.

    For each component (default: attention, mlp), captures its output during
    `prompt_a`, then patches that output into the forward pass on `prompt_b`,
    and measures the fraction of the metric gap that the patch closes.

    `backend_hint` is accepted but unused (this function uses module hooks
    directly via `resolve_subcomponent_module`).
    """
    del backend_hint   # unused; kept for API symmetry

    with torch.no_grad():
        out_a = model(**prompt_a, return_dict=True)
        out_b = model(**prompt_b, return_dict=True)
    metric_a = metric_fn(out_a)
    metric_b = metric_fn(out_b)
    total_gap = metric_a - metric_b

    contributions: dict[str, float] = {}
    for comp in components:
        # 1) Capture component outputs during prompt_a.
        capture_hooks = []
        src_acts_by_layer: dict[int, list] = {}
        for idx in layer_indices:
            module = resolve_subcomponent_module(model, idx, comp)
            if module is None:
                continue
            captured: list = []

            def capture(mod, inp, out, store=captured):
                store.append(out[0] if isinstance(out, tuple) else out)
            capture_hooks.append(module.register_forward_hook(capture))
            src_acts_by_layer[idx] = captured

        try:
            with torch.no_grad():
                model(**prompt_a, return_dict=True)
        finally:
            _remove_hooks(capture_hooks)

        # 2) Patch captured outputs into prompt_b's forward pass.
        patch_hooks = []
        for idx in layer_indices:
            if idx not in src_acts_by_layer:
                continue
            module = resolve_subcomponent_module(model, idx, comp)
            if module is None:
                continue
            stored = src_acts_by_layer[idx]
            if not stored:
                continue
            captured_out = stored[0]

            def patch(mod, inp, out, repl=captured_out):
                if isinstance(out, tuple):
                    return (repl,) + out[1:]
                return repl
            patch_hooks.append(module.register_forward_hook(patch))

        try:
            with torch.no_grad():
                patched_out = model(**prompt_b, return_dict=True)
            patched_metric = metric_fn(patched_out)
        finally:
            _remove_hooks(patch_hooks)

        contributions[comp] = float((patched_metric - metric_b) / (total_gap + 1e-9))

    return DIMResult(
        components=contributions,
        total=float(total_gap),
        layer_range=(min(layer_indices), max(layer_indices)),
    )
