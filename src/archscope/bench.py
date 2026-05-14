"""InterpBench — standardized mechanistic interpretability benchmark.

Run a fixed test suite on any model to get a comparable "interp profile".
Designed to support cross-architecture comparisons (transformer, hybrid, SSM).

Tests:
1. probe_sentiment_auroc   — Can we linearly probe pos/neg sentiment?
2. probe_math_auroc        — Can we linearly probe math vs non-math?
3. induction_head          — Does the model copy A->B in-context?
4. copy_circuit            — Can it copy a word list verbatim?
5. concentration           — How peaked are next-token predictions?
6. sae_dictionary_quality  — Dense SAE reconstruction at mid-layer
7. ssm_state_info          — (Mamba only) variance ratio of SSM state across inputs

Output: dataclass `InterpProfile` with all scores, JSON-serializable.
A model's "InterpProfile" is its interp signature.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import torch

from . import probes, sae, circuits
from .backends import Backend


@dataclass
class InterpProfile:
    """Standardized interp test results for one model."""
    model_name: str
    arch_family: str            # "transformer" | "hybrid" | "ssm" | "custom"
    n_params: int
    n_layers: int
    hidden_dim: int

    # Probe tasks (in-arch AUROC at representative mid-layer)
    probe_sentiment_auroc: float = float("nan")
    probe_math_auroc: float = float("nan")

    # Circuit detection (relative to baseline)
    induction_head_relative: float = float("nan")
    copy_accuracy: float = float("nan")
    concentration_relative: float = float("nan")

    # SAE quality (dense SAE recon at mid-layer)
    sae_dense_recon: float = float("nan")
    sae_rank1_recon: float = float("nan")
    sae_better: str = "—"   # "dense" or "rank1"

    # SSM-specific (NaN for non-SSM)
    ssm_state_variance_ratio: float = float("nan")

    runtime_seconds: float = 0.0
    notes: list = field(default_factory=list)


# ---------- Probe tasks ----------

SENTIMENT_POS = [
    "I love this movie, it's amazing!", "What a wonderful day today.",
    "Best book I have ever read.", "She is so kind and thoughtful.",
    "The food was absolutely delicious.", "I'm thrilled about the great news.",
    "Such a beautiful sunset tonight.", "He always makes me laugh.",
    "Amazing performance overall.", "Truly a delightful experience.",
    "Incredible work, congratulations!", "I admire her dedication.",
    "The concert was breathtaking.", "He is the kindest person I know.",
    "This was pure joy from start.", "I had a wonderful evening.",
]
SENTIMENT_NEG = [
    "I hate this place absolutely.", "What a terrible movie that was.",
    "This is the worst day ever.", "She is mean and selfish.",
    "The food was awful and cold.", "I'm so disappointed with everything.",
    "Such a horrible meeting today.", "He always annoys me deeply.",
    "What an awful experience overall.", "I despise everything about this.",
    "This restaurant is dreadful.", "She is the rudest person here.",
    "The concert was a disaster.", "He never listens to anyone properly.",
    "Truly a miserable evening.", "I regret coming here entirely.",
]
MATH = [
    "Solve for x: 2x + 3 = 11.", "Compute the derivative of x cubed.",
    "The integral from 0 to 1 of x dx.", "Triangle with angles 30 60 90.",
    "The eigenvalue of matrix M is.", "By chain rule d/dx of sin x squared.",
    "Find roots of x squared minus 5x plus 6.", "The Taylor series of e to x.",
    "Prove sum of two evens is even.", "The Cauchy-Schwarz inequality.",
    "Let f be continuous on closed interval.", "Define the limit as n grows large.",
    "By induction on the natural numbers.", "The dot product of u and v.",
    "A group is abelian if commutative.", "Fundamental theorem of calculus.",
]
NONMATH = [
    "The cat sat on the mat softly.", "Music has the power to move us.",
    "Mountains stretch to the horizon.", "She whispered her secret carefully.",
    "Birds sing at dawn every morning.", "The chef prepared dinner slowly.",
    "Children laughed in the city park.", "Rain pattered against the window.",
    "Coffee aroma filled the kitchen.", "Time passes faster when busy.",
    "Stars appeared in the dark sky.", "The river flowed gently downstream.",
    "Snow blanketed the entire valley.", "Dance is a universal language.",
    "Books contain endless worlds.", "Travel broadens one's perspective.",
]


def _run_probe(model, backend_hint, tokenize_fn, pos_texts, neg_texts, layer_name):
    inputs_pos = tokenize_fn(pos_texts)
    inputs_neg = tokenize_fn(neg_texts)
    pf = probes.fit_probe(
        model, inputs_pos=inputs_pos, inputs_neg=inputs_neg,
        layer_name=layer_name, backend_hint=backend_hint,
    )
    # Use train AUROC (val with tiny n is too noisy)
    return float(pf.metrics["train_auroc"])


def _run_sae(model, backend, layer_name, hidden_dim, tokenize_fn):
    """Train both Dense and Rank-1 SAE, return both recon errors."""
    corpus = SENTIMENT_POS + SENTIMENT_NEG + MATH + NONMATH
    inputs = tokenize_fn(corpus)
    with torch.no_grad():
        rec = backend.extract(inputs, layers=[layer_name])[0]
    acts = rec.activations.reshape(-1, hidden_dim).detach()
    cfg_d = sae.SAEConfig(input_dim=hidden_dim, n_features=512, sae_type="dense",
                           sparsity=1e-4, learning_rate=3e-3)
    cfg_r = sae.SAEConfig(input_dim=hidden_dim, n_features=512, sae_type="rank1",
                           sparsity=1e-4, learning_rate=3e-3)
    sae_d = sae.fit_sae(acts, cfg_d, epochs=60)
    sae_r = sae.fit_sae(acts, cfg_r, epochs=60)
    return sae_d.last_metrics["recon"], sae_r.last_metrics["recon"]


def _ssm_state_variance_ratio(model, backend, layer_name, tokenize_fn) -> float:
    """For Mamba: how much variance does the SSM state have across diverse inputs?

    Ratio = variance across inputs / total variance. High = state encodes input strongly.
    """
    corpus = SENTIMENT_POS[:8] + MATH[:8]   # diverse inputs
    inputs = tokenize_fn(corpus)
    rec = backend.extract(inputs, layers=[layer_name])[0]
    # rec.activations shape: (B, intermediate, ssm_state)
    flat = rec.activations.reshape(rec.activations.shape[0], -1)
    var_across = flat.var(dim=0).mean().item()
    var_total = flat.var().item()
    return float(var_across / (var_total + 1e-9))


# ---------- Main runner ----------

def benchmark(
    model_name: str,
    model,
    tokenizer,
    backend_hint: str,
    arch_family: str = "transformer",
    tokenize_fn=None,
    sentiment_layer: int | None = None,
    math_layer: int | None = None,
    sae_layer: int | None = None,
    ssm_layer: int | None = None,
) -> InterpProfile:
    """Run InterpBench on a model. Returns InterpProfile.

    `tokenize_fn(texts: list[str]) -> dict` is the model-specific tokenizer wrapper.
    """
    backend = Backend.for_model(model, hint=backend_hint)
    t_start = time.time()

    len(backend.layer_names())
    # Filter to only residual layers if mixed (e.g., Mamba has both .residual and .ssm_state)
    residual_layers = [n for n in backend.layer_names() if ".residual" in n]
    n_blocks = len(residual_layers)
    hidden_dim = backend.hidden_dim(residual_layers[0])
    n_params = sum(p.numel() for p in model.parameters())

    profile = InterpProfile(
        model_name=model_name,
        arch_family=arch_family,
        n_params=n_params,
        n_layers=n_blocks,
        hidden_dim=hidden_dim,
    )

    # Choose representative layers: sentiment probes at shallow depth, math
    # and SAE at mid-depth. Each can be overridden by the caller.
    if sentiment_layer is None:
        sentiment_layer = n_blocks // 4
    if math_layer is None:
        math_layer = n_blocks // 2
    if sae_layer is None:
        sae_layer = n_blocks // 2

    try:
        profile.probe_sentiment_auroc = _run_probe(
            model, backend_hint, tokenize_fn,
            SENTIMENT_POS, SENTIMENT_NEG,
            f"layer_{sentiment_layer}.residual",
        )
    except Exception as e:
        profile.notes.append(f"probe_sentiment error: {str(e)[:60]}")

    try:
        profile.probe_math_auroc = _run_probe(
            model, backend_hint, tokenize_fn,
            MATH, NONMATH,
            f"layer_{math_layer}.residual",
        )
    except Exception as e:
        profile.notes.append(f"probe_math error: {str(e)[:60]}")

    # Circuit detection
    try:
        circs = circuits.run_all_circuits(model, tokenizer=tokenizer, device="cpu")
        profile.induction_head_relative = circs["induction_head"].relative
        if "copy_circuit" in circs:
            profile.copy_accuracy = circs["copy_circuit"].score
        if "early_token_concentration" in circs:
            profile.concentration_relative = circs["early_token_concentration"].relative
    except Exception as e:
        profile.notes.append(f"circuits error: {str(e)[:60]}")

    # SAE quality
    try:
        d_recon, r_recon = _run_sae(
            model, backend, f"layer_{sae_layer}.residual", hidden_dim, tokenize_fn,
        )
        profile.sae_dense_recon = d_recon
        profile.sae_rank1_recon = r_recon
        profile.sae_better = "rank1" if r_recon < d_recon else "dense"
    except Exception as e:
        profile.notes.append(f"sae error: {str(e)[:60]}")

    # SSM state (mamba/ssm only)
    if arch_family == "ssm" and ssm_layer is not None:
        try:
            profile.ssm_state_variance_ratio = _ssm_state_variance_ratio(
                model, backend, f"layer_{ssm_layer}.ssm_state", tokenize_fn,
            )
        except Exception as e:
            profile.notes.append(f"ssm_state error: {str(e)[:60]}")

    profile.runtime_seconds = time.time() - t_start
    return profile


def profile_to_markdown(profile: InterpProfile) -> str:
    """Format profile as markdown row for the leaderboard."""
    lines = [
        f"### {profile.model_name}",
        f"  Arch: {profile.arch_family} | Params: {profile.n_params/1e6:.1f}M | "
        f"Layers: {profile.n_layers} | Hidden: {profile.hidden_dim}",
        "  | Test | Score |",
        "  |------|-------|",
        f"  | Sentiment probe AUROC    | {profile.probe_sentiment_auroc:.3f} |",
        f"  | Math probe AUROC          | {profile.probe_math_auroc:.3f} |",
        f"  | Induction head (×chance)  | {profile.induction_head_relative:>.1f} |",
        f"  | Copy accuracy             | {profile.copy_accuracy:.2%} |",
        f"  | Concentration (rel)       | {profile.concentration_relative:.3f} |",
        f"  | SAE Dense recon           | {profile.sae_dense_recon:.4f} |",
        f"  | SAE Rank-1 recon          | {profile.sae_rank1_recon:.4f} |",
        f"  | SAE better                | {profile.sae_better} |",
        f"  | SSM state var ratio       | {profile.ssm_state_variance_ratio:.3f} |",
        f"  | Runtime                   | {profile.runtime_seconds:.1f}s |",
    ]
    if profile.notes:
        lines.append(f"  Notes: {'; '.join(profile.notes)}")
    return "\n".join(lines)
