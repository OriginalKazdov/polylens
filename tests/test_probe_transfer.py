"""Probe transfer experiment: Pythia → Kazdov via paired-activation alignment.

This is the punch line of the workshop paper. Question:
  Does a probe trained on Pythia activations transfer to Kazdov via linear
  alignment, or are the representations arch-specific?

Result interpretation:
  - transfer_auroc ≈ baseline_target → REPRESENTATIONS ARE ALIGNED (probe transfers)
  - transfer_auroc near 0.5 → REPRESENTATIONS ARE ARCH-SPECIFIC (no transfer)
  - Both are publishable.
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/polylens/src")

from polylens import transfer
from polylens.backends import Backend
from polylens.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"
PYTHIA_NAME = "EleutherAI/pythia-160m"


# ---------- Texts ----------

ALIGN_TEXTS = [   # used to learn the M alignment (NO LABELS NEEDED)
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

MATH_TRAIN = [
    "Solve for y: 3y = 12.",
    "Differentiate sin x squared.",
    "Evaluate integral of 1 over x.",
    "Find the limit of x squared at 2.",
    "Triangle has sides 3 4 5.",
    "Determinant of identity matrix.",
    "Sum of geometric series.",
    "Prove pythagorean theorem.",
]
NONMATH_TRAIN = [
    "The dog ran across the meadow.",
    "She wrote a letter to her friend.",
    "The sun set behind the mountains.",
    "He enjoys reading mystery novels.",
    "The library was quiet today.",
    "Rain fell softly all afternoon.",
    "The artist painted with great care.",
    "Children built sandcastles on the beach.",
]

MATH_TEST = [
    "Find x in 5x equals 25.",
    "Compute derivative of log x.",
    "Integral of cos x from 0 to pi.",
    "The Taylor expansion of cosine.",
    "Solve the quadratic equation.",
    "Find the eigenvalue of A.",
    "Prove that 2 plus 2 equals 4.",
    "The matrix product distributes over.",
]
NONMATH_TEST = [
    "The river flowed quietly past.",
    "Birds returned from migration in spring.",
    "She baked a delicious chocolate cake.",
    "The orchestra played a moving piece.",
    "Stars filled the clear winter sky.",
    "He told a story by the fire.",
    "The garden bloomed with colorful flowers.",
    "Waves crashed against the rocky shore.",
]


def tokenize_hf(tokenizer, texts):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


def tokenize_kazdov(tokenizer, texts):
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}


def main():
    t_start = time.time()

    # --- Load both models
    print(f"[setup] Loading Pythia-160m…")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pythia_tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if pythia_tok.pad_token is None: pythia_tok.pad_token = pythia_tok.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, dtype=torch.float32)
    pythia_model.eval()

    print(f"[setup] Loading kazdov-α…")
    kazdov_model, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)

    pythia_backend = Backend.for_model(pythia_model, hint="transformer")
    kazdov_backend = Backend.for_model(kazdov_model, hint="kazdov")

    print(f"  Pythia d=768, Kazdov d=512")
    print(f"  Align texts: {len(ALIGN_TEXTS)}, Train: {len(MATH_TRAIN)}+{len(NONMATH_TRAIN)}, Test: {len(MATH_TEST)}+{len(NONMATH_TEST)}")
    print()

    # --- Sweep across layers
    print("="*78)
    print("PROBE TRANSFER: Pythia → Kazdov (math vs non-math task)")
    print("="*78)
    print(f"\n{'L_py':>5} {'L_kz':>5} | {'src_AUROC':>9} {'tgt_AUROC':>9} {'TRANSFER':>9} {'drop':>6}")
    print(f"{'-'*5} {'-'*5}-+-{'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    results = []
    # Test transfer at a few representative layers
    layer_pairs = [
        (2, 2), (5, 5), (8, 8), (11, 11),  # matched depth
        (5, 2), (5, 8), (5, 11),            # cross-depth
    ]
    for py_l, kz_l in layer_pairs:
        try:
            r = transfer.evaluate_transfer(
                source_model=pythia_model,
                target_model=kazdov_model,
                source_backend=pythia_backend,
                target_backend=kazdov_backend,
                source_tokenize=lambda t: tokenize_hf(pythia_tok, t),
                target_tokenize=lambda t: tokenize_kazdov(kazdov_tok, t),
                align_texts=ALIGN_TEXTS,
                train_pos=MATH_TRAIN, train_neg=NONMATH_TRAIN,
                test_pos=MATH_TEST,   test_neg=NONMATH_TEST,
                source_layer=f"layer_{py_l}.residual",
                target_layer=f"layer_{kz_l}.residual",
                source_arch_name="transformer",
                target_arch_name="kazdov",
            )
            print(f"{py_l:>5} {kz_l:>5} | {r.baseline_source_auroc:>9.3f} {r.baseline_target_auroc:>9.3f} {r.transfer_auroc:>9.3f} {r.transfer_drop:>+6.3f}")
            results.append((py_l, kz_l, r))
        except Exception as e:
            print(f"{py_l:>5} {kz_l:>5} | ERROR: {e}")

    # --- Now reverse direction: Kazdov → Pythia
    print()
    print("="*78)
    print("PROBE TRANSFER: Kazdov → Pythia (math vs non-math task)")
    print("="*78)
    print(f"\n{'L_kz':>5} {'L_py':>5} | {'src_AUROC':>9} {'tgt_AUROC':>9} {'TRANSFER':>9} {'drop':>6}")
    print(f"{'-'*5} {'-'*5}-+-{'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    for kz_l, py_l in [(2, 2), (5, 5), (8, 8), (11, 11)]:
        try:
            r = transfer.evaluate_transfer(
                source_model=kazdov_model,
                target_model=pythia_model,
                source_backend=kazdov_backend,
                target_backend=pythia_backend,
                source_tokenize=lambda t: tokenize_kazdov(kazdov_tok, t),
                target_tokenize=lambda t: tokenize_hf(pythia_tok, t),
                align_texts=ALIGN_TEXTS,
                train_pos=MATH_TRAIN, train_neg=NONMATH_TRAIN,
                test_pos=MATH_TEST,   test_neg=NONMATH_TEST,
                source_layer=f"layer_{kz_l}.residual",
                target_layer=f"layer_{py_l}.residual",
                source_arch_name="kazdov",
                target_arch_name="transformer",
            )
            print(f"{kz_l:>5} {py_l:>5} | {r.baseline_source_auroc:>9.3f} {r.baseline_target_auroc:>9.3f} {r.transfer_auroc:>9.3f} {r.transfer_drop:>+6.3f}")
        except Exception as e:
            print(f"{kz_l:>5} {py_l:>5} | ERROR: {e}")

    print(f"\n{'='*78}")
    print(f"Runtime: {time.time()-t_start:.1f}s")
    print(f"{'='*78}")
    print("\nINTERPRETATION:")
    print("  transfer_AUROC ≈ tgt_AUROC → probe DIRECTIONS align (features cross-arch)")
    print("  transfer_AUROC ≈ 0.5      → probe DIRECTIONS do NOT align (arch-specific)")
    print("  transfer_drop = tgt - transfer (smaller = better alignment)")


if __name__ == "__main__":
    main()
