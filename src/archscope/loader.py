"""High-level model loading helper.

Eliminates ~5 lines of HuggingFace boilerplate per example:

    # Before
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    backend = Backend.for_model(model, hint=arch)

    # After
    model, tok, backend = archscope.load_model(name, arch="transformer")
"""
from __future__ import annotations
from typing import Any

from .backends import Backend


def load_model(
    name: str,
    arch: str | None = None,
    dtype: Any = None,
    device: str = "cpu",
) -> tuple[Any, Any, Backend]:
    """Load a HuggingFace model + tokenizer + matching backend in one call.

    Args:
        name: HuggingFace model id (e.g., "EleutherAI/pythia-160m") OR a path
            to a local archscope-registered model (e.g., kazdov checkpoint).
        arch: Backend hint — one of "transformer", "mamba", "kazdov",
            "recurrent". If None, auto-detect from HF config.model_type.
        dtype: torch dtype. Defaults to torch.float32.
        device: device string. Default "cpu".

    Returns:
        (model, tokenizer, backend) ready for use in probes/sae/lens/etc.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model = model.to(device).eval()

    backend = Backend.for_model(model, hint=arch)
    return model, tokenizer, backend


def make_tokenize_fn(tokenizer, max_length: int = 32, attention_mask_bool: bool = False):
    """Return a tokenize function suitable for ``Backend.extract`` and ``probes.fit_probe``.

    Args:
        tokenizer: a HuggingFace tokenizer.
        max_length: max sequence length to truncate to.
        attention_mask_bool: if True, returns attention_mask as a bool tensor
            (required for ``kazdov`` backend); HF default is int64.

    Returns:
        A callable ``texts -> dict`` that matches the ``inputs`` format used
        across archscope.
    """
    def fn(texts):
        out = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)
        if attention_mask_bool and "attention_mask" in out:
            out["attention_mask"] = out["attention_mask"].bool()
        return out
    return fn
