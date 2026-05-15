"""Logit lens and Tuned lens — project intermediate residual stream into vocab space.

The logit lens (Nostalgebraist, 2020) applies the model's own final norm + unembedding
to every layer's residual stream, revealing "what the model would predict if forced
to commit at this layer." It tells you which layers do the work of forming the final
prediction.

The tuned lens (Belrose et al, 2023) learns per-layer affine transformations that
correct for representation drift, producing a more faithful intermediate decoding.

Both work on any architecture exposing residual stream via Backend.extract — i.e.
transformer, mamba, kazdov, custom recurrent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .backends import Backend
from ._utils import resolve_unembedding, resolve_final_norm


@dataclass
class LayerPrediction:
    """What the lens says at one layer."""
    layer: int
    layer_name: str
    top_tokens: list[tuple[int, str, float]]    # (token_id, token_str, prob)
    target_prob: float | None = None             # if target_token provided, prob at this layer
    target_rank: int | None = None               # rank of target token in this layer's distribution
    entropy: float = 0.0                          # confidence (low = peaked, high = uncertain)


@dataclass
class LensResult:
    """Output of a lens pass over multiple layers."""
    prompt: str
    target_token: str | None = None
    target_token_id: int | None = None
    layers: list[LayerPrediction] = field(default_factory=list)
    method: str = "logit_lens"

    def to_markdown(self) -> str:
        """Quick formatted display."""
        lines = [f"### {self.method} on `{self.prompt}`"]
        if self.target_token:
            lines.append(f"Target: `{self.target_token}` (id={self.target_token_id})")
        lines.append("")
        lines.append("| Layer | top-1 token | top-1 prob | target prob | rank | entropy |")
        lines.append("|-------|-------------|-----------:|------------:|-----:|--------:|")
        for lp in self.layers:
            if lp.top_tokens:
                top_id, top_str, top_p = lp.top_tokens[0]
            else:
                top_str, top_p = "?", float("nan")
            tgt_p = f"{lp.target_prob:.3f}" if lp.target_prob is not None else "—"
            tgt_r = f"{lp.target_rank}" if lp.target_rank is not None else "—"
            lines.append(
                f"| {lp.layer:>2} | `{top_str[:20]}` | {top_p:.3f} | {tgt_p} | {tgt_r} | {lp.entropy:.3f} |"
            )
        return "\n".join(lines)


def _logits_to_pred(
    logits: torch.Tensor,
    tokenizer,
    target_id: int | None = None,
    top_k: int = 5,
) -> tuple[list[tuple[int, str, float]], float, int | None, float]:
    """Convert logit vector → (top_k_tokens, target_prob, target_rank, entropy)."""
    probs = torch.softmax(logits.float(), dim=-1)
    top_p, top_i = probs.topk(top_k)
    top_tokens = []
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        try:
            tok_str = tokenizer.decode([i])
        except Exception:
            tok_str = f"<id={i}>"
        top_tokens.append((i, tok_str, p))
    if target_id is not None:
        target_prob = float(probs[target_id].item())
        target_rank = int((probs > probs[target_id]).sum().item())
    else:
        target_prob = None
        target_rank = None
    # Shannon entropy in nats
    entropy = float(-(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log()).sum().item())
    return top_tokens, target_prob, target_rank, entropy


def logit_lens(
    model,
    tokenizer,
    prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    backend_hint: str | None = None,
    top_k: int = 5,
    device: str = "cpu",
) -> LensResult:
    """Apply model's own final-norm + unembedding to each layer's residual stream.

    Returns per-layer predictions: top-k tokens + entropy + (optional) target rank/prob.

    The classic question this answers: "at which layer does the model commit to the
    final token?" Often you'll see entropy decrease and target rank climb across layers.
    """
    backend = Backend.for_model(model, hint=backend_hint)
    norm = resolve_final_norm(model)
    unembed = resolve_unembedding(model)
    if unembed is None:
        raise ValueError("Could not locate model's lm_head / unembedding module.")

    # Tokenize prompt (handle both HF tokenizer + kazdov-style inputs)
    enc = tokenizer(prompt, return_tensors="pt")
    if hasattr(enc, "input_ids"):
        inputs = {"input_ids": enc.input_ids.to(device)}
    else:
        inputs = {"input_ids": enc["input_ids"].to(device)}

    target_id = None
    if target_token is not None:
        target_ids = tokenizer(target_token, add_special_tokens=False).input_ids
        target_id = target_ids[0] if target_ids else None

    # Pick layers
    all_layer_names = [n for n in backend.layer_names() if ".residual" in n]
    if layers is None:
        layer_names = all_layer_names
    else:
        layer_names = [f"layer_{i}.residual" for i in layers if f"layer_{i}.residual" in all_layer_names]

    # Extract residual streams
    with torch.no_grad():
        records = backend.extract(inputs, layers=layer_names)

    result = LensResult(
        prompt=prompt,
        target_token=target_token,
        target_token_id=target_id,
        method="logit_lens",
    )

    for rec in records:
        # rec.activations shape: (B, T, hidden) — take last-token residual
        last = rec.activations[:, -1, :]   # (B, hidden)
        # Apply final norm if present (transformer family). For kazdov it's `model.ln_f`.
        if norm is not None:
            try:
                last = norm(last)
            except Exception:
                pass
        # Project to vocab
        logits = unembed(last)[0]   # (vocab,)
        top_tokens, tgt_p, tgt_r, ent = _logits_to_pred(logits, tokenizer, target_id, top_k)
        idx = int(rec.layer_name.split("_")[1].split(".")[0])
        result.layers.append(LayerPrediction(
            layer=idx, layer_name=rec.layer_name,
            top_tokens=top_tokens,
            target_prob=tgt_p, target_rank=tgt_r,
            entropy=ent,
        ))

    return result


# -------------------- TUNED LENS --------------------

class TunedLens(nn.Module):
    """Learned per-layer affine transformations into the unembedding space.

    For each layer, learn a transformation: residual → adjusted_residual that, when
    fed through the model's final_norm + unembedding, better matches the final-layer
    distribution. Following Belrose et al 2023.

    Usage:
        tl = TunedLens.fit(model, tokenizer, calibration_texts, backend_hint="transformer")
        result = tl.predict(model, tokenizer, prompt="...", backend_hint="transformer")
    """

    def __init__(self, model, n_layers: int, hidden_dim: int, backend_hint: str | None = None):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.backend_hint = backend_hint
        # One Linear per layer; identity initialization (so untrained tuned lens ≈ logit lens)
        self.translators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            for _ in range(n_layers)
        ])
        for t in self.translators:
            nn.init.eye_(t.weight)
            nn.init.zeros_(t.bias)
        self._model_ref = [model]   # avoid registering as submodule

    @classmethod
    def fit(
        cls,
        model,
        tokenizer,
        calibration_texts: list[str],
        backend_hint: str | None = None,
        epochs: int = 30,
        lr: float = 1e-3,
        max_len: int = 64,
        device: str = "cpu",
    ) -> "TunedLens":
        """Train per-layer translators to match the model's own final distribution."""
        backend = Backend.for_model(model, hint=backend_hint)
        layer_names = [n for n in backend.layer_names() if ".residual" in n]
        n_layers = len(layer_names)
        # Get hidden dim from first layer
        hidden_dim = backend.hidden_dim(layer_names[0])

        tl = cls(model, n_layers=n_layers, hidden_dim=hidden_dim, backend_hint=backend_hint).to(device)
        norm = resolve_final_norm(model)
        unembed = resolve_unembedding(model)

        opt = torch.optim.AdamW(tl.translators.parameters(), lr=lr)

        # Pre-extract all activations + target logits once.
        # Ensure tokenizer has a pad token (GPT-2 family ships without one).
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        enc = tokenizer(calibration_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len)
        inputs = {"input_ids": enc["input_ids"].to(device)}
        if "attention_mask" in enc:
            inputs["attention_mask"] = enc["attention_mask"].to(device)

        # Per-row index of the last REAL (non-pad) token. If no attention_mask
        # (single, unpadded sequence), the conventional last-position is fine.
        if "attention_mask" in enc:
            real_lengths = enc["attention_mask"].sum(dim=1).to(device)  # (B,)
            last_idx = (real_lengths - 1).clamp(min=0)
        else:
            B = inputs["input_ids"].shape[0]
            last_idx = torch.full((B,), inputs["input_ids"].shape[1] - 1,
                                   dtype=torch.long, device=device)

        def gather_last(acts: torch.Tensor) -> torch.Tensor:
            # acts: (B, T, H) → (B, H) at each row's real last position.
            B = acts.shape[0]
            return acts[torch.arange(B, device=acts.device), last_idx]

        with torch.no_grad():
            records = backend.extract(inputs, layers=layer_names)
            # Target: model's actual final logits at last REAL position per row.
            final_residual = gather_last(records[-1].activations)
            if norm is not None:
                final_residual = norm(final_residual)
            target_logits = unembed(final_residual).detach()    # (B, vocab)
            target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)

        for epoch in range(epochs):
            opt.zero_grad()
            total_loss = 0.0
            for i, rec in enumerate(records):
                last = gather_last(rec.activations).detach()
                translated = tl.translators[i](last)
                if norm is not None:
                    translated = norm(translated)
                pred_logits = unembed(translated)
                pred_log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
                # KL divergence target_log_probs || pred_log_probs
                loss = torch.nn.functional.kl_div(pred_log_probs, target_log_probs,
                                                   log_target=True, reduction="batchmean")
                total_loss = total_loss + loss
            total_loss.backward()
            opt.step()

        tl.last_loss = float(total_loss.item() / max(n_layers, 1))
        return tl

    def predict(
        self,
        model,
        tokenizer,
        prompt: str,
        target_token: str | None = None,
        backend_hint: str | None = None,
        layers: list[int] | None = None,
        top_k: int = 5,
        device: str = "cpu",
    ) -> LensResult:
        """Apply the trained tuned lens to a new prompt."""
        backend = Backend.for_model(model, hint=backend_hint or self.backend_hint)
        norm = resolve_final_norm(model)
        unembed = resolve_unembedding(model)

        enc = tokenizer(prompt, return_tensors="pt")
        inputs = {"input_ids": enc["input_ids"].to(device)}
        target_id = None
        if target_token is not None:
            ids = tokenizer(target_token, add_special_tokens=False).input_ids
            target_id = ids[0] if ids else None

        all_layer_names = [n for n in backend.layer_names() if ".residual" in n]
        if layers is None:
            layer_names = all_layer_names
        else:
            layer_names = [f"layer_{i}.residual" for i in layers if f"layer_{i}.residual" in all_layer_names]

        with torch.no_grad():
            records = backend.extract(inputs, layers=layer_names)

        result = LensResult(prompt=prompt, target_token=target_token,
                            target_token_id=target_id, method="tuned_lens")

        for rec in records:
            idx = int(rec.layer_name.split("_")[1].split(".")[0])
            last = rec.activations[:, -1, :]
            with torch.no_grad():
                translated = self.translators[idx](last)
                if norm is not None:
                    translated = norm(translated)
                logits = unembed(translated)[0]
            top_tokens, tgt_p, tgt_r, ent = _logits_to_pred(logits, tokenizer, target_id, top_k)
            result.layers.append(LayerPrediction(
                layer=idx, layer_name=rec.layer_name,
                top_tokens=top_tokens,
                target_prob=tgt_p, target_rank=tgt_r,
                entropy=ent,
            ))
        return result
