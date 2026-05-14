"""Internal utilities — not part of the public API.

Shared layer-resolution logic used by neurons.py and attribute.py to find
the actual nn.Module corresponding to a layer_name string like
"layer_5.residual" across different HuggingFace model architectures.
"""
from __future__ import annotations
from typing import Any


# Common HF architecture paths to layer ModuleList.
# Order matters: first match wins. Add new architectures here.
_LAYER_PATHS: list[tuple[str, str | None]] = [
    ("model", "layers"),         # Llama / Mistral / Qwen / GPT-NeoX-style
    ("transformer", "h"),         # GPT-2 / Falcon
    ("transformer", "blocks"),    # MPT
    ("gpt_neox", "layers"),       # Pythia
    ("backbone", "layers"),       # Mamba / Mamba-2
    ("layers", None),             # Direct .layers (some custom models, e.g. kazdov)
    ("h", None),                  # Direct .h
    ("blocks", None),             # Direct .blocks (kazdov)
]


def _parse_layer_index(layer_name: str) -> int | None:
    """Extract the integer index from a name like 'layer_5.residual'."""
    try:
        idx_part = layer_name.split("_")[1].split(".")[0]
        return int(idx_part)
    except (IndexError, ValueError):
        return None


def resolve_layer_module(model: Any, layer_name: str):
    """Return the nn.Module corresponding to a layer_name across HF naming conventions.

    Handles: Llama, Mistral, Qwen, GPT-2, Falcon, MPT, Pythia, Mamba, custom .blocks.

    Returns None if the layer name cannot be parsed or no path matches.
    """
    idx = _parse_layer_index(layer_name)
    if idx is None:
        return None
    for parent_attr, child_attr in _LAYER_PATHS:
        parent_obj = getattr(model, parent_attr, None)
        if parent_obj is None:
            continue
        layers = parent_obj if child_attr is None else getattr(parent_obj, child_attr, None)
        if layers is None:
            continue
        try:
            return layers[idx]
        except (IndexError, TypeError):
            continue
    return None


_UNEMBED_PATHS = [
    "lm_head",        # Llama, Pythia, Mistral, Mamba, kazdov, most HF CausalLMs
    "embed_out",      # some HF models
    "output_layer",   # some custom models
]

_FINAL_NORM_PATHS: list[tuple[str, str | None]] = [
    ("model", "norm"),                  # Llama / Mistral
    ("gpt_neox", "final_layer_norm"),    # Pythia
    ("transformer", "ln_f"),             # GPT-2 / Falcon
    ("backbone", "norm_f"),              # Mamba
    ("ln_f", None),                      # kazdov (top-level)
]


def resolve_unembedding(model: Any):
    """Find the model's unembedding / lm_head module. Returns nn.Module or None."""
    for path in _UNEMBED_PATHS:
        m = getattr(model, path, None)
        if m is not None:
            return m
    return None


def resolve_final_norm(model: Any):
    """Find the model's final pre-unembedding layer norm. Returns module or None."""
    for parent_attr, child_attr in _FINAL_NORM_PATHS:
        parent_obj = getattr(model, parent_attr, None)
        if parent_obj is None:
            continue
        norm = parent_obj if child_attr is None else getattr(parent_obj, child_attr, None)
        if norm is not None:
            return norm
    return None


def resolve_subcomponent_module(model: Any, idx: int, component: str):
    """Find an attention or MLP submodule inside a layer at index `idx`.

    component: "attention" or "mlp".
    Returns None if the component isn't found.
    """
    layer = resolve_layer_module(model, f"layer_{idx}.residual")
    if layer is None:
        return None
    if component == "attention":
        for attr in ("self_attn", "attn", "attention"):
            sub = getattr(layer, attr, None)
            if sub is not None:
                return sub
    elif component == "mlp":
        for attr in ("mlp", "feed_forward", "ffn"):
            sub = getattr(layer, attr, None)
            if sub is not None:
                return sub
    return None
