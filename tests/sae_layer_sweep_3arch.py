"""SAE comparison layer-by-layer across 3 architectures.

For each layer of each architecture, train both Dense and Rank-1 SAEs and
compare reconstruction error. The hypothesis: rank-1 (WriteSAE-style)
factorization should help on architectures with bilinear/recurrent structure
(Kazdov hybrid, Mamba SSM) and hurt on pure attention (Pythia).

This is the layer-by-layer detail behind our headline architectural finding.
"""
from __future__ import annotations
import sys
import time
import json
import os
import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/archscope/src")

from archscope import sae
from archscope.backends import Backend
from archscope.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"
PYTHIA_NAME = "EleutherAI/pythia-160m"
MAMBA_NAME = "state-spaces/mamba-130m-hf"


# Diverse training corpus (~40 sentences across topics for SAE training data)
SAE_CORPUS = [
    "The cat sat on the mat softly today.",
    "Solve for x: 2x + 3 = 11 equation.",
    "Music has the power to move us.",
    "Compute the derivative of x cubed.",
    "Mountains stretch to the distant horizon.",
    "The integral from 0 to 1 of x dx.",
    "Children laughed in the city park.",
    "Find roots of x squared minus 5x plus 6.",
    "Coffee aroma filled the morning kitchen.",
    "The Cauchy-Schwarz inequality states clearly.",
    "Snow blanketed the entire mountain valley.",
    "Let f be a continuous function on the interval.",
    "Dance is a universal language of joy.",
    "By induction on the natural numbers.",
    "The chef prepared a delicate dinner slowly.",
    "Define the limit as n approaches infinity.",
    "Travel broadens one's overall perspective.",
    "A group is abelian if and only if commutative.",
    "Books contain endless imagined worlds.",
    "The fundamental theorem of calculus connects.",
    "The dog ran across the meadow.",
    "She wrote a letter to her friend.",
    "The sun set behind the mountains tonight.",
    "He enjoys reading mystery novels every evening.",
    "The library was quiet today.",
    "Rain fell softly all afternoon.",
    "The artist painted with great care.",
    "Children built sandcastles on the beach.",
    "Find x in 5x equals 25.",
    "Compute derivative of log x.",
    "Integral of cos x from 0 to pi.",
    "The Taylor expansion of cosine.",
    "Solve the quadratic equation.",
    "Find the eigenvalue of A.",
    "Prove that 2 plus 2 equals 4.",
    "The matrix product distributes over.",
    "The river flowed quietly past.",
    "Birds returned from migration in spring.",
    "She baked a delicious chocolate cake.",
    "The orchestra played a moving piece.",
]


def tokenize_hf(tokenizer, texts):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


def tokenize_kazdov(tokenizer, texts):
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}


def fit_both_saes(activations, hidden_dim, n_features=512, epochs=60):
    """Train Dense + Rank-1 SAE on same activations. Returns (dense_recon, rank1_recon)."""
    cfg_d = sae.SAEConfig(input_dim=hidden_dim, n_features=n_features, sae_type="dense",
                           sparsity=1e-4, learning_rate=3e-3)
    cfg_r = sae.SAEConfig(input_dim=hidden_dim, n_features=n_features, sae_type="rank1",
                           sparsity=1e-4, learning_rate=3e-3)
    sae_d = sae.fit_sae(activations, cfg_d, epochs=epochs)
    sae_r = sae.fit_sae(activations, cfg_r, epochs=epochs)
    return sae_d.last_metrics["recon"], sae_r.last_metrics["recon"]


def sweep_arch(name, model, backend, tokenize_fn, hidden_dim, n_layers):
    print(f"\n  Architecture: {name} ({n_layers} layers, d={hidden_dim})")
    print(f"  {'Layer':>5} | {'Dense':>8} | {'Rank-1':>8} | {'better':>6} | {'ratio':>6}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    results = []
    inputs = tokenize_fn(SAE_CORPUS)
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx}.residual"
        try:
            rec = backend.extract(inputs, layers=[layer_name])[0]
            acts = rec.activations.reshape(-1, hidden_dim).detach()
            dense_r, rank1_r = fit_both_saes(acts, hidden_dim)
            better = "rank-1" if rank1_r < dense_r else "dense"
            ratio = rank1_r / dense_r if dense_r > 0 else float("inf")
            print(f"  {layer_idx:>5} | {dense_r:>8.4f} | {rank1_r:>8.4f} | {better:>6} | {ratio:>6.2f}")
            results.append({
                "layer": layer_idx, "dense_recon": dense_r, "rank1_recon": rank1_r,
                "better": better, "ratio_r1_over_d": ratio,
            })
        except Exception as e:
            print(f"  {layer_idx:>5} | ERROR: {str(e)[:60]}")
    return results


def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[setup] Loading 3 models…")
    pythia_tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if pythia_tok.pad_token is None: pythia_tok.pad_token = pythia_tok.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, dtype=torch.float32)
    pythia_model.eval()

    mamba_tok = AutoTokenizer.from_pretrained(MAMBA_NAME)
    if mamba_tok.pad_token is None: mamba_tok.pad_token = mamba_tok.eos_token
    mamba_model = AutoModelForCausalLM.from_pretrained(MAMBA_NAME, dtype=torch.float32)
    mamba_model.eval()

    kazdov_model, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)

    print("="*82)
    print("SAE LAYER SWEEP — Dense vs Rank-1 reconstruction across 3 architectures")
    print("="*82)
    print("\nHypothesis: Rank-1 SAE outperforms Dense on architectures with bilinear")
    print("or recurrent state structure; underperforms on pure attention.")

    all_results = {}

    all_results["pythia"] = sweep_arch(
        "Pythia (standard transformer)",
        pythia_model,
        Backend.for_model(pythia_model, hint="transformer"),
        lambda t: tokenize_hf(pythia_tok, t),
        pythia_model.config.hidden_size,
        pythia_model.config.num_hidden_layers,
    )

    all_results["kazdov"] = sweep_arch(
        "kazdov-α (hybrid MoBE-BCN+MHA)",
        kazdov_model,
        Backend.for_model(kazdov_model, hint="kazdov"),
        lambda t: tokenize_kazdov(kazdov_tok, t),
        kazdov_model.d_model,
        len(kazdov_model.blocks),
    )

    all_results["mamba"] = sweep_arch(
        "Mamba-130m (true SSM)",
        mamba_model,
        Backend.for_model(mamba_model, hint="mamba"),
        lambda t: tokenize_hf(mamba_tok, t),
        mamba_model.config.hidden_size,
        mamba_model.config.num_hidden_layers,
    )

    # Summary
    print()
    print("="*82)
    print("SUMMARY — fraction of layers where Rank-1 BEATS Dense")
    print("="*82)
    print(f"\n{'Arch':<15} | {'Layers':>7} | {'rank-1 wins':>12} | {'mean ratio':>10}")
    print(f"{'-'*15}-+-{'-'*7}-+-{'-'*12}-+-{'-'*10}")
    for arch_name, results in all_results.items():
        n = len(results)
        rank1_wins = sum(1 for r in results if r.get("better") == "rank-1")
        ratios = [r["ratio_r1_over_d"] for r in results if "ratio_r1_over_d" in r]
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        print(f"{arch_name:<15} | {n:>7} | {rank1_wins}/{n:>10} | {mean_ratio:>10.3f}")
    print("\n(ratio < 1.0 means rank-1 has lower reconstruction error)")

    # Save
    out_path = "/Users/kazdov/code/OriginalKazdov/archscope/_research/sae_layer_sweep_3arch.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"\n{'='*82}\nRuntime: {time.time()-t_start:.1f}s\n{'='*82}")


if __name__ == "__main__":
    main()
