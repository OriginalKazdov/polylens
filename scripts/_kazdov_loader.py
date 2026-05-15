"""Internal-only loader for kazdov-α checkpoints.

NOT shipped to PyPI. This depends on a private model implementation
(``~/code/OriginalKazdov/kazdov/``) and is only useful for archscope's
maintainer to validate kazdov-family experiments against the rest of
the library.

Anyone else with a custom architecture should write their own loader
and pass the resulting model to ``Backend.for_model(model, hint="kazdov")``.
See ``src/archscope/kazdov_backend.py`` for what the backend expects.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch


KAZDOV_REPO = Path.home() / "code" / "OriginalKazdov" / "kazdov"


def _ensure_kazdov_importable() -> None:
    """Make the private kazdov package importable if its repo is on disk."""
    p = str(KAZDOV_REPO)
    if not KAZDOV_REPO.exists():
        raise FileNotFoundError(
            f"kazdov repo not found at {KAZDOV_REPO}. This loader is internal "
            f"to archscope's maintainer. Anyone else with a custom architecture "
            f"should write their own loader and call Backend.for_model(model, "
            f"hint='kazdov')."
        )
    if p not in sys.path:
        sys.path.insert(0, p)


def load_kazdov_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    """Load kazdov-α from a checkpoint directory.

    Expects: ``config.json`` + ``final.pt`` (or ``latest.pt``) in the directory.
    Returns: ``(model in eval mode, GPT-2 tokenizer)``.
    """
    _ensure_kazdov_importable()
    from kazdov.kazdov_lm import KazdovLM   # type: ignore

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

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
