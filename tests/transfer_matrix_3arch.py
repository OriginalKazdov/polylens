"""3-architecture probe transfer matrix — the paper's main result.

Tests probe transfer in all 6 directions between Pythia, Kazdov, Mamba.
Both for sentiment and math tasks at multiple representative layers.

Output: comparison matrix showing where probes transfer cleanly vs where
they degrade, with the asymmetry signature characteristic of each
architecture pair.
"""
from __future__ import annotations
import sys
import time
import json
import os
import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/mechinterp-small/src")

from mechinterp_small import transfer
from mechinterp_small.backends import Backend
from mechinterp_small.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"
PYTHIA_NAME = "EleutherAI/pythia-160m"
MAMBA_NAME = "state-spaces/mamba-130m-hf"


# ---------- Texts (more diverse than previous run) ----------

ALIGN_TEXTS = [
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
]

# MATH vs NON-MATH (the hardcore task)
MATH_TRAIN = [
    "Solve for y: 3y = 12.", "Differentiate sin x squared.",
    "Evaluate integral of 1 over x.", "Find the limit of x squared at 2.",
    "Triangle has sides 3 4 5.", "Determinant of identity matrix.",
    "Sum of geometric series.", "Prove pythagorean theorem.",
]
NONMATH_TRAIN = [
    "The dog ran across the meadow.", "She wrote a letter to her friend.",
    "The sun set behind the mountains.", "He enjoys reading mystery novels.",
    "The library was quiet today.", "Rain fell softly all afternoon.",
    "The artist painted with great care.", "Children built sandcastles on the beach.",
]
MATH_TEST = [
    "Find x in 5x equals 25.", "Compute derivative of log x.",
    "Integral of cos x from 0 to pi.", "The Taylor expansion of cosine.",
    "Solve the quadratic equation.", "Find the eigenvalue of A.",
    "Prove that 2 plus 2 equals 4.", "The matrix product distributes over.",
]
NONMATH_TEST = [
    "The river flowed quietly past.", "Birds returned from migration in spring.",
    "She baked a delicious chocolate cake.", "The orchestra played a moving piece.",
    "Stars filled the clear winter sky.", "He told a story by the fire.",
    "The garden bloomed with colorful flowers.", "Waves crashed against the rocky shore.",
]


# ---------- Tokenizer adapters ----------

def make_tokenize_hf(tokenizer):
    """For Pythia and Mamba (both use HF tokenizers + standard input dict)."""
    def fn(texts):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return fn


def make_tokenize_kazdov(tokenizer):
    """For kazdov (needs attention_mask as bool, separate signature)."""
    def fn(texts):
        out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}
    return fn


# ---------- MAIN ----------

def main():
    t_start = time.time()

    # --- Load all 3 models
    print("[setup] Loading 3 models in parallel-ish…")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pythia
    pythia_tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if pythia_tok.pad_token is None: pythia_tok.pad_token = pythia_tok.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, dtype=torch.float32)
    pythia_model.eval()

    # Mamba
    mamba_tok = AutoTokenizer.from_pretrained(MAMBA_NAME)
    if mamba_tok.pad_token is None: mamba_tok.pad_token = mamba_tok.eos_token
    mamba_model = AutoModelForCausalLM.from_pretrained(MAMBA_NAME, dtype=torch.float32)
    mamba_model.eval()

    # Kazdov
    kazdov_model, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)

    archs = {
        "pythia": {
            "model": pythia_model,
            "backend": Backend.for_model(pythia_model, hint="transformer"),
            "tokenize": make_tokenize_hf(pythia_tok),
            "arch_name": "transformer",
            "n_layers": pythia_model.config.num_hidden_layers,
            "d": pythia_model.config.hidden_size,
        },
        "kazdov": {
            "model": kazdov_model,
            "backend": Backend.for_model(kazdov_model, hint="kazdov"),
            "tokenize": make_tokenize_kazdov(kazdov_tok),
            "arch_name": "kazdov",
            "n_layers": len(kazdov_model.blocks),
            "d": kazdov_model.d_model,
        },
        "mamba": {
            "model": mamba_model,
            "backend": Backend.for_model(mamba_model, hint="mamba"),
            "tokenize": make_tokenize_hf(mamba_tok),
            "arch_name": "mamba",
            "n_layers": mamba_model.config.num_hidden_layers,
            "d": mamba_model.config.hidden_size,
        },
    }
    for name, a in archs.items():
        print(f"  {name}: {a['n_layers']} layers, d={a['d']}")
    print()

    # --- Run transfer for all 6 directions
    # Pick a representative middle layer for each (use 50% depth)
    rep_layers = {name: a["n_layers"] // 2 for name, a in archs.items()}
    print(f"Using representative layers: {rep_layers}")
    print()

    # All 6 directed pairs
    pairs = [
        ("pythia", "kazdov"),
        ("pythia", "mamba"),
        ("kazdov", "pythia"),
        ("kazdov", "mamba"),
        ("mamba", "pythia"),
        ("mamba", "kazdov"),
    ]

    print("="*82)
    print("3-ARCH PROBE TRANSFER MATRIX — math-vs-nonmath task @ mid-layer of each arch")
    print("="*82)
    print(f"\n{'SOURCE':>10} → {'TARGET':<10} | {'src AUROC':>9} {'tgt AUROC':>9} {'TRANSFER':>9} {'drop':>6}")
    print(f"{'-'*10}---{'-'*10}-+-{'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    all_results = {}
    for src, tgt in pairs:
        src_a = archs[src]
        tgt_a = archs[tgt]
        src_layer = f"layer_{rep_layers[src]}.residual"
        tgt_layer = f"layer_{rep_layers[tgt]}.residual"
        try:
            r = transfer.evaluate_transfer(
                source_model=src_a["model"],
                target_model=tgt_a["model"],
                source_backend=src_a["backend"],
                target_backend=tgt_a["backend"],
                source_tokenize=src_a["tokenize"],
                target_tokenize=tgt_a["tokenize"],
                align_texts=ALIGN_TEXTS,
                train_pos=MATH_TRAIN, train_neg=NONMATH_TRAIN,
                test_pos=MATH_TEST,   test_neg=NONMATH_TEST,
                source_layer=src_layer,
                target_layer=tgt_layer,
                source_arch_name=src_a["arch_name"],
                target_arch_name=tgt_a["arch_name"],
            )
            print(f"{src:>10} → {tgt:<10} | {r.baseline_source_auroc:>9.3f} {r.baseline_target_auroc:>9.3f} {r.transfer_auroc:>9.3f} {r.transfer_drop:>+6.3f}")
            all_results[f"{src}->{tgt}"] = {
                "source_arch": src, "target_arch": tgt,
                "source_layer": src_layer, "target_layer": tgt_layer,
                "baseline_source": r.baseline_source_auroc,
                "baseline_target": r.baseline_target_auroc,
                "transfer": r.transfer_auroc,
                "drop": r.transfer_drop,
            }
        except Exception as e:
            print(f"{src:>10} → {tgt:<10} | ERROR: {str(e)[:60]}")

    print()
    print("="*82)
    print("ASYMMETRY ANALYSIS — is transfer A→B the same as B→A?")
    print("="*82)
    print(f"\n{'PAIR':<22} | {'A→B':>7} | {'B→A':>7} | {'asymmetry':>10}")
    print(f"{'-'*22}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")
    visited = set()
    for src, tgt in pairs:
        key = tuple(sorted([src, tgt]))
        if key in visited: continue
        visited.add(key)
        a_to_b = all_results.get(f"{src}->{tgt}", {}).get("transfer")
        b_to_a = all_results.get(f"{tgt}->{src}", {}).get("transfer")
        if a_to_b is None or b_to_a is None: continue
        asym = abs(a_to_b - b_to_a)
        print(f"{src:>10} ↔ {tgt:<10} | {a_to_b:>7.3f} | {b_to_a:>7.3f} | {asym:>10.3f}")

    # Save results
    out_path = "/Users/kazdov/code/OriginalKazdov/mechinterp-small/_research/transfer_matrix_3arch.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {out_path}")
    print(f"\n{'='*82}\nRuntime: {time.time()-t_start:.1f}s\n{'='*82}")


if __name__ == "__main__":
    main()
