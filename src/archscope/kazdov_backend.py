"""Backend for custom architectures that expose layers via ``model.blocks``.

Originally written for kazdov-α (a transformer-style decoder LM with hybrid
MoBE-BCN + MHA attention) — but the backend is generic. It works for ANY
PyTorch model where:

- residual blocks are exposed as ``model.blocks`` (a ``nn.ModuleList``)
- ``model.d_model`` (or ``model.hidden_size``) is set on the model
- forward signature is ``model(input_ids, attention_mask=None, ...)``

This is the simplest pattern for registering a custom architecture with
archscope. If your model uses a different convention (e.g., ``model.layers``
under another parent), subclass ``Backend`` directly — this module is a
working example.

The backend registers under the name ``"kazdov"`` for historical reasons.
It used to be coupled to a private model-loading function; that function
was moved out of the shipped package since it depended on a private
repository. To load your own custom model, do it yourself and then call
``Backend.for_model(model, hint="kazdov")``.
"""
from __future__ import annotations
import torch

from .backends import Backend, ActivationRecord


@Backend.register("kazdov")
class KazdovBackend(Backend):
    """Generic backend for models exposing layers via ``model.blocks``.

    Captures the output of each block via forward hooks (the model is
    expected to not implement ``output_hidden_states=True`` natively).

    Requirements on the model:
    - ``model.blocks`` is a ``nn.ModuleList`` of residual blocks.
    - ``model.d_model`` or ``model.hidden_size`` is set.
    - ``model(input_ids, attention_mask=...)`` is the forward signature.
    """

    def layer_names(self) -> list[str]:
        n_layers = len(self.model.blocks)
        return [f"layer_{i}.residual" for i in range(n_layers)]

    def extract(self, inputs, layers=None):
        layers = layers or self.layer_names()
        self._validate_layers(layers)
        captures: dict[str, torch.Tensor] = {}

        # Register a forward hook on each requested block.
        hooks = []
        for layer_name in layers:
            idx = int(layer_name.split("_")[1].split(".")[0])
            if idx >= len(self.model.blocks):
                continue
            block = self.model.blocks[idx]

            def make_hook(name):
                def hook(module, inp, out):
                    tensor = out if isinstance(out, torch.Tensor) else out[0]
                    captures[name] = tensor.detach()
                return hook
            hooks.append(block.register_forward_hook(make_hook(layer_name)))

        try:
            with torch.no_grad():
                if isinstance(inputs, dict):
                    input_ids = inputs["input_ids"]
                    attn = inputs.get("attention_mask")
                else:
                    input_ids = inputs
                    attn = None
                self.model(input_ids, attention_mask=attn)
        finally:
            for h in hooks:
                h.remove()

        records = []
        for layer_name in layers:
            if layer_name not in captures:
                continue
            records.append(ActivationRecord(
                layer_name=layer_name,
                activations=captures[layer_name],
                meta={"kind": "residual", "arch": "kazdov-blocks"},
            ))
        return records

    def hidden_dim(self, layer_name: str) -> int:
        # Some custom models expose this as `d_model`, others as `hidden_size`.
        for attr in ("d_model", "hidden_size"):
            v = getattr(self.model, attr, None)
            if v is not None:
                return v
        raise ValueError(
            f"Cannot infer hidden_dim for {type(self.model).__name__}: "
            f"set model.d_model or model.hidden_size, or subclass KazdovBackend "
            f"and override hidden_dim()."
        )
