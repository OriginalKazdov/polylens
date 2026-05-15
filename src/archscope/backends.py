"""Architecture-agnostic activation extraction.

The core abstraction: a `Backend` knows how to hook into a model and pull out
hidden states at named layers, regardless of underlying framework
(PyTorch/JAX/custom).

Three backends implemented:
- TransformerBackend: HuggingFace transformers (residual stream per layer)
- MambaBackend: state-space models (hidden state + ssm state per layer)
- RecurrentBackend: generic RNN-like (extracts hidden state per timestep)

Custom architectures (e.g., kazdov MoBE-BCN) register via Backend.register().
"""
from __future__ import annotations
import abc
import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class ActivationRecord:
    """Captured activations from a single forward pass.

    Attributes:
        layer_name: identifier of the layer (e.g., "blocks.5.residual")
        activations: tensor of shape (batch, seq_len, hidden_dim) typically
        meta: arch-specific metadata (e.g., {'kind': 'residual'} or {'kind': 'ssm_state'})
    """
    layer_name: str
    activations: Any   # torch.Tensor or jax.Array
    meta: dict


class Backend(abc.ABC):
    """Abstract interface — extract activations from any model architecture."""

    _registry: dict[str, type["Backend"]] = {}

    @classmethod
    def register(cls, name: str):
        def deco(klass):
            cls._registry[name] = klass
            return klass
        return deco

    # HF model_type → backend name. Transformer family covers most HF decoder LMs;
    # add new families here as they ship. Auto-detect intentionally raises when
    # nothing matches (silent fallback caused real bugs in v0.2.4).
    _AUTODETECT = {
        # transformer family
        "llama":       "transformer",
        "mistral":     "transformer",
        "qwen2":       "transformer",
        "qwen3":       "transformer",
        "gpt2":        "transformer",
        "gpt_neox":    "transformer",   # Pythia uses gpt_neox
        "gpt_neo":     "transformer",
        "gptj":        "transformer",
        "falcon":      "transformer",
        "mpt":         "transformer",
        "bloom":       "transformer",
        "opt":         "transformer",
        "phi":         "transformer",
        "phi3":        "transformer",
        "gemma":       "transformer",
        "gemma2":      "transformer",
        "starcoder2":  "transformer",
        # SSM family
        "mamba":       "mamba",
        "mamba2":      "mamba",
    }

    @classmethod
    def for_model(cls, model: Any, hint: str | None = None) -> "Backend":
        """Auto-detect (or use hint) to select a backend.

        Raises ValueError if no hint is provided and the model's ``config.model_type``
        is not in the autodetect table. Pass ``hint=...`` explicitly for any model
        that's not auto-detected, or register a custom backend via
        ``Backend.register('name')``.
        """
        if hint:
            if hint in cls._registry:
                return cls._registry[hint](model)
            raise ValueError(
                f"Unknown backend hint '{hint}'. Registered: {sorted(cls._registry)}"
            )
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type in cls._AUTODETECT:
            return cls._registry[cls._AUTODETECT[model_type]](model)
        raise ValueError(
            f"No backend matches model with config.model_type={model_type!r} "
            f"(type {type(model).__name__}). Pass hint=... explicitly, or "
            f"register a custom backend via Backend.register('name'). "
            f"Auto-detected types: {sorted(cls._AUTODETECT)}"
        )

    def __init__(self, model: Any):
        self.model = model

    @abc.abstractmethod
    def layer_names(self) -> list[str]:
        """Return list of layer identifiers we can hook."""
        ...

    @abc.abstractmethod
    def extract(self, inputs: Any, layers: list[str] | None = None) -> list[ActivationRecord]:
        """Run forward pass, return activations at requested layers (all if None)."""
        ...

    @abc.abstractmethod
    def hidden_dim(self, layer_name: str) -> int:
        """Dimensionality of activations at a given layer."""
        ...

    def _validate_layers(self, layers: list[str]) -> None:
        """Raise a clear error if any requested layer name isn't valid."""
        valid = set(self.layer_names())
        bad = [ln for ln in layers if ln not in valid]
        if bad:
            # Show first few valid examples so users see the format.
            sample = ", ".join(self.layer_names()[:3])
            n_total = len(valid)
            raise ValueError(
                f"Unknown layer name(s) for {type(self).__name__}: {bad}. "
                f"Valid layer names look like: {sample}{', ...' if n_total > 3 else ''} "
                f"(total: {n_total} layer names). Call `backend.layer_names()` "
                f"to see all valid options."
            )


@Backend.register("transformer")
class TransformerBackend(Backend):
    """HuggingFace transformers backend — extracts residual stream per layer."""

    def layer_names(self) -> list[str]:
        # Layer names are virtual handles consumed by .extract(), which uses
        # HF's `output_hidden_states=True` to retrieve the residual stream
        # (no direct attribute walk into model.model.layers[i] needed —
        # so this works across HF decoder LM families).
        n_layers = getattr(self.model.config, "num_hidden_layers", 0)
        return [f"layer_{i}.residual" for i in range(n_layers)]

    def extract(self, inputs, layers=None):
        layers = layers or self.layer_names()
        self._validate_layers(layers)
        # Use HF's output_hidden_states=True for clean extraction.
        # Wrap in no_grad: extraction shouldn't build a backward graph.
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        records = []
        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
        for layer_name in layers:
            # Parse "layer_N.residual" → N+1 (since [0] is embedding output)
            idx = int(layer_name.split("_")[1].split(".")[0]) + 1
            records.append(ActivationRecord(
                layer_name=layer_name,
                activations=hidden_states[idx],
                meta={"kind": "residual", "arch": "transformer"},
            ))
        return records

    def hidden_dim(self, layer_name: str) -> int:
        return self.model.config.hidden_size


@Backend.register("mamba")
class MambaBackend(Backend):
    """Mamba/Mamba-2 backend — extracts residual stream AND SSM recurrent state.

    Works with HuggingFace MambaForCausalLM (and Mamba2ForCausalLM).

    Two flavors of activations exposed:
    - `layer_N.residual`  → residual stream after block N (B, T, hidden_size)
    - `layer_N.ssm_state` → final SSM recurrent state after processing the
                            sequence at block N: shape (B, intermediate_size, ssm_state_size).
                            This exposes the recurrent state used by Mamba-style models —
                            useful when experiments need access to memory-like state rather
                            than residual activations alone.
    """

    def layer_names(self) -> list[str]:
        n_layers = getattr(self.model.config, "n_layer", 0) or getattr(self.model.config, "num_hidden_layers", 0)
        out = []
        for i in range(n_layers):
            out.append(f"layer_{i}.residual")
            out.append(f"layer_{i}.ssm_state")
        return out

    def extract(self, inputs, layers=None):
        layers = layers or self.layer_names()
        self._validate_layers(layers)

        need_residual = any(".residual" in ln for ln in layers)
        need_ssm = any(".ssm_state" in ln for ln in layers)

        with torch.no_grad():
            if need_ssm:
                # Pass a DynamicCache so Mamba writes final SSM states to it
                from transformers.models.mamba.modeling_mamba import DynamicCache
                cache = DynamicCache(config=self.model.config)
                outputs = self.model(
                    **inputs,
                    cache_params=cache,
                    use_cache=True,
                    output_hidden_states=need_residual,
                    return_dict=True,
                )
            else:
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                cache = None

        records = []
        for layer_name in layers:
            idx = int(layer_name.split("_")[1].split(".")[0])
            if ".residual" in layer_name:
                records.append(ActivationRecord(
                    layer_name=layer_name,
                    activations=outputs.hidden_states[idx + 1].detach(),  # +1 because [0] is embedding output
                    meta={"kind": "residual", "arch": "mamba", "shape_meaning": "(B, T, hidden_size)"},
                ))
            elif ".ssm_state" in layer_name:
                if cache is None or not hasattr(cache, "layers") or idx >= len(cache.layers):
                    continue
                ssm = cache.layers[idx].recurrent_states
                if ssm is None:
                    continue
                records.append(ActivationRecord(
                    layer_name=layer_name,
                    activations=ssm.detach(),
                    meta={
                        "kind": "ssm_state",
                        "arch": "mamba",
                        "shape_meaning": "(B, intermediate_size, ssm_state_size)",
                        "d_inner": ssm.shape[-2],
                        "d_state": ssm.shape[-1],
                    },
                ))
        return records

    def hidden_dim(self, layer_name: str) -> int:
        if ".ssm_state" in layer_name:
            # SSM state is (intermediate_size × ssm_state_size)
            d_inner = getattr(self.model.config, "intermediate_size", None)
            d_state = getattr(self.model.config, "state_size", None)
            if d_inner and d_state:
                return d_inner * d_state
            # Fallback: introspect from a block
            mixer = self.model.backbone.layers[0].mixer
            return mixer.intermediate_size * mixer.ssm_state_size
        return getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)


@Backend.register("recurrent")
class RecurrentBackend(Backend):
    """Generic recurrent backend — for custom RNN-family models (e.g., kazdov MoBE-BCN).

    Expects model to expose:
    - .get_hidden_states(inputs) → dict[str, tensor]
    OR registered forward hooks (user injects).

    Custom models should subclass and override `extract`.
    """

    def layer_names(self) -> list[str]:
        # Default: try common RNN attribute names
        if hasattr(self.model, "n_layer"):
            return [f"layer_{i}.hidden" for i in range(self.model.n_layer)]
        if hasattr(self.model, "num_layers"):
            return [f"layer_{i}.hidden" for i in range(self.model.num_layers)]
        return ["layer_0.hidden"]

    def extract(self, inputs, layers=None):
        # Generic — subclass should override
        if hasattr(self.model, "get_hidden_states"):
            hs = self.model.get_hidden_states(inputs)
            return [
                ActivationRecord(layer_name=k, activations=v, meta={"kind": "hidden", "arch": "recurrent"})
                for k, v in hs.items()
            ]
        raise NotImplementedError(
            f"RecurrentBackend default extract() not implemented for {type(self.model).__name__}. "
            "Subclass and override extract(), or call model.get_hidden_states() yourself."
        )

    def hidden_dim(self, layer_name: str) -> int:
        for attr in ("d_model", "hidden_size", "d_hidden", "n_embd"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        raise ValueError("Cannot infer hidden_dim — override hidden_dim() in subclass.")
