"""Cross-architecture circuit detection.

Classical mech-interp circuits were discovered in transformers (induction heads,
attention sinks, copy circuits). It's an OPEN QUESTION whether they exist in
SSMs (Mamba) or hybrid attention (Kazdov-α).

Each detector is BEHAVIORAL — measures the model's outputs given crafted
inputs — so it works on ANY architecture without per-arch internals.

Detectors implemented:
- induction_head_score: does the model copy A->B associations? (Olsson et al 2022)
- copy_score: can the model copy a sequence verbatim?
- early_token_attention: proxy for attention-sink behavior, measures prob mass
  concentrated on early tokens (BOS-like).
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CircuitScore:
    """Score from one circuit test."""
    name: str
    score: float                # primary metric, higher = circuit more present
    baseline: float             # random-model baseline
    relative: float             # score / baseline (>1 means circuit present)
    raw: dict                   # detailed numbers


# ---------- Helper: model-agnostic forward ----------

def _model_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Call model and extract final-step logits. Works for HF + kazdov."""
    with torch.no_grad():
        out = model(input_ids)
    if isinstance(out, dict):
        logits = out.get("logits", out.get("last_hidden_state"))
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    return logits


# ---------- Induction head detection ----------

def induction_head_score(
    model,
    n_pairs: int = 20,
    seq_len: int = 6,
    n_trials: int = 50,
    vocab_size: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> CircuitScore:
    """Olsson-style induction head test.

    Construct a sequence: [A1 B1] [A2 B2] ... [A_k B_k] [A_1]
    Score = probability that next token is B_1 (the original pair completion).

    If induction heads exist, the model should learn to copy A_1 → B_1 in-context.
    Compared to random-token baseline (1/vocab_size).
    """
    rng = random.Random(seed)
    if vocab_size is None:
        # Try to infer (HF transformers + custom models like kazdov)
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        elif hasattr(model, "vocab_size"):
            vocab_size = model.vocab_size
        else:
            vocab_size = 50257   # GPT-2 default

    successes = 0
    rank_sum = 0.0
    prob_target_sum = 0.0
    for trial in range(n_trials):
        # Pick n_pairs random token pairs
        tokens = rng.sample(range(100, min(vocab_size, 40000)), 2 * n_pairs)
        seq = []
        pairs = []
        for i in range(n_pairs):
            a, b = tokens[2*i], tokens[2*i+1]
            seq.extend([a, b])
            pairs.append((a, b))
        # Append cue: A1 — model should predict B1
        a1, b1 = pairs[0]
        seq.append(a1)

        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = _model_logits(model, ids)
        # last position prediction
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        # Did the model predict b1 as top-1?
        pred = int(torch.argmax(last_logits).item())
        if pred == b1:
            successes += 1
        # Rank of b1 in the logit distribution (0 = predicted top-1).
        rank = int((last_logits > last_logits[b1]).sum().item())
        rank_sum += rank
        prob_target_sum += float(probs[b1].item())

    accuracy = successes / n_trials
    avg_rank = rank_sum / n_trials
    avg_prob = prob_target_sum / n_trials
    chance = 1.0 / vocab_size
    return CircuitScore(
        name="induction_head",
        score=avg_prob,
        baseline=chance,
        relative=avg_prob / chance,
        raw={
            "accuracy_top1": accuracy,
            "avg_rank_target": avg_rank,
            "avg_prob_target": avg_prob,
            "n_trials": n_trials,
            "seq_len_pairs": n_pairs,
        },
    )


# ---------- Copy circuit detection ----------

def copy_score(
    model,
    tokenizer,
    n_trials: int = 30,
    n_words: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> CircuitScore:
    """Test whether model can copy a short word-list verbatim after a separator.

    Input pattern: "list: A B C D E. list: " → predict "A"
                   then after predicting A, predict B, etc.

    Score = fraction of correctly-copied tokens.
    """
    rng = random.Random(seed)
    word_pool = [
        "cat", "dog", "tree", "moon", "star", "fish", "bird", "rock",
        "leaf", "wave", "sky", "rain", "snow", "wind", "fire", "ice",
        "book", "lamp", "key", "door", "wall", "road", "hill", "sand",
    ]

    correct = 0
    total = 0
    for trial in range(n_trials):
        words = rng.sample(word_pool, n_words)
        prompt = f"list: {' '.join(words)}. list: "
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        ids.shape[1]

        # Token IDs for target words (first token of each)
        target_tokens = []
        for w in words:
            target_tokens.append(tokenizer(" " + w, add_special_tokens=False).input_ids[0])

        # Autoregressively predict n_words tokens, chaining the model's own
        # predictions (not teacher-forcing) — measures cumulative copy ability.
        cur = ids.clone()
        for tgt in target_tokens:
            logits = _model_logits(model, cur)
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            if next_tok == tgt:
                correct += 1
            total += 1
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)

    acc = correct / total if total > 0 else 0.0
    # Random baseline for copying is ~1/vocab_size; effectively ~0
    vocab_size = (getattr(model, "vocab_size", None)
                  or getattr(getattr(model, "config", None), "vocab_size", 50257))
    chance = 1.0 / vocab_size
    return CircuitScore(
        name="copy_circuit",
        score=acc,
        baseline=chance,
        relative=acc / chance if chance > 0 else float("inf"),
        raw={"n_trials": n_trials, "n_words": n_words, "correct": correct, "total": total},
    )


# ---------- Early-token attention proxy (attention sink) ----------

def early_token_attention(
    model,
    tokenizer,
    texts: list[str] | None = None,
    device: str = "cpu",
) -> CircuitScore:
    """Behavioral proxy for "attention sink" — measures how much logit mass
    concentrates on the first token's vocab prediction when given a continuation cue.

    Construct: "[BOS] X1 X2 ... X_n [predict ?]"
    Measure: entropy of next-token distribution.

    Low entropy → model is highly concentrated (possible sink-like behavior).
    High entropy → no concentration.

    NOTE: this is a coarse behavioral proxy. True attention-sink analysis
    requires architecture-specific attention weights. But the proxy works
    on ANY architecture (including SSMs that don't have attention).
    """
    if texts is None:
        texts = [
            "The cat sat on the", "Music has the power to",
            "Mountains stretch to the", "She wrote a letter to",
            "The sun set behind the", "Children laughed in the",
            "Solve for x in the equation", "Compute the derivative of the",
        ]
    entropies = []
    for txt in texts:
        ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
        logits = _model_logits(model, ids)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        # Shannon entropy (in nats)
        ent = -(probs * (probs.clamp(min=1e-12).log())).sum().item()
        entropies.append(ent)

    avg_ent = sum(entropies) / len(entropies)
    # Reference: log(vocab_size) is max entropy
    vocab_size = (getattr(model, "vocab_size", None)
                  or getattr(getattr(model, "config", None), "vocab_size", 50257))
    max_ent = torch.log(torch.tensor(float(vocab_size))).item()
    return CircuitScore(
        name="early_token_concentration",
        score=avg_ent,             # in nats
        baseline=max_ent,
        relative=avg_ent / max_ent,  # 0 = full concentration, 1 = uniform
        raw={"per_text_entropy": entropies, "max_entropy": max_ent},
    )


# ---------- Runner ----------

def run_all_circuits(model, tokenizer=None, device: str = "cpu") -> dict[str, CircuitScore]:
    """Run all available circuit tests on a model.

    Returns a dict keyed by circuit name. Some tests require tokenizer; if not
    provided, those are skipped.
    """
    results = {}
    # Induction head only needs model + vocab_size
    results["induction_head"] = induction_head_score(model, device=device)
    if tokenizer is not None:
        results["copy_circuit"] = copy_score(model, tokenizer, device=device)
        results["early_token_concentration"] = early_token_attention(model, tokenizer, device=device)
    return results
