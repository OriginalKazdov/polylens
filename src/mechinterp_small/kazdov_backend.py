"""Backend for kazdov-α (and related Kazdov family models).

Kazdov-α is a transformer-style decoder LM with hybrid attention (MoBE-BCN
mixture of bilinear experts + standard MHA in parallel). Architecturally
closer to standard transformer than to pure RNN/SSM — but the BCN attention
branch makes it a distinct architecture family for cross-arch interp.

Differences from HF transformer:
- No HF AutoModelForCausalLM interface (custom forward signature)
- Layers exposed as `model.blocks` (ModuleList)
- No `output_hidden_states=True` argument — we capture via forward hooks
- Forward signature: (input_ids, attention_mask=None, labels=None)
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

from .backends import Backend, ActivationRecord


KAZDOV_REPO = Path.home() / "code" / "OriginalKazdov" / "kazdov"


def _ensure_kazdov_importable():
    """Add kazdov repo to sys.path so we can import KazdovLM."""
    p = str(KAZDOV_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_kazdov_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    """Load kazdov-α from a checkpoint directory.

    Expects: config.json + final.pt (or latest.pt) in the directory.
    Returns: (model in eval mode, tokenizer wrapper).
    """
    _ensure_kazdov_importable()
    from kazdov.kazdov_lm import KazdovLM
    import json

    ckpt_dir = Path(checkpoint_path)
    config = json.loads((ckpt_dir / "config.json").read_text())
    model_cfg = config["model_cfg"]

    model = KazdovLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        rank=model_cfg["rank"],
        mlp_dim=model_cfg.get("mlp_dim"),
        max_len=model_cfg.get("max_len", 256),
        use_trilinear=model_cfg.get("use_trilinear", False),
        use_bi_bcn=model_cfg.get("use_bi_bcn", False),
        use_hybrid_mha=model_cfg.get("use_hybrid_mha", True),
        use_mobe=model_cfg.get("use_mobe", False),
        n_experts=model_cfg.get("n_experts", 1),
    )

    # Try final.pt then latest.pt
    for fname in ("final.pt", "latest.pt"):
        f = ckpt_dir / fname
        if f.exists():
            state = torch.load(f, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            break
    else:
        raise FileNotFoundError(f"No final.pt or latest.pt in {ckpt_dir}")

    model.to(device).eval()

    # Tokenizer: kazdov used GPT-2 tokenizer per memory
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@Backend.register("kazdov")
class KazdovBackend(Backend):
    """Backend for kazdov-family models (KazdovLM, MoBE-BCN variants).

    Uses forward hooks to capture residual stream after each KazdovBlock,
    since the model doesn't expose output_hidden_states.
    """

    def layer_names(self) -> list[str]:
        n_layers = len(self.model.blocks)
        return [f"layer_{i}.residual" for i in range(n_layers)]

    def extract(self, inputs, layers=None):
        layers = layers or self.layer_names()
        captures: dict[str, torch.Tensor] = {}

        # Register hooks on each requested block
        hooks = []
        for layer_name in layers:
            idx = int(layer_name.split("_")[1].split(".")[0])
            if idx >= len(self.model.blocks): continue
            block = self.model.blocks[idx]

            def make_hook(name):
                def h(module, inp, out):
                    captures[name] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
                return h
            hooks.append(block.register_forward_hook(make_hook(layer_name)))

        try:
            # Forward — kazdov signature: model(input_ids, attention_mask=None)
            with torch.no_grad():
                input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
                attn = inputs.get("attention_mask") if isinstance(inputs, dict) else None
                self.model(input_ids, attention_mask=attn)
        finally:
            for h in hooks: h.remove()

        records = []
        for layer_name in layers:
            if layer_name not in captures: continue
            records.append(ActivationRecord(
                layer_name=layer_name,
                activations=captures[layer_name],
                meta={"kind": "residual", "arch": "kazdov-mobe-bcn"},
            ))
        return records

    def hidden_dim(self, layer_name: str) -> int:
        return self.model.d_model
