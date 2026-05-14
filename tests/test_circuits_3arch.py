"""Run circuit detectors across all 3 architectures.

Generates the paper's main table: which circuits exist in transformer, hybrid,
and SSM architectures, with quantitative scores.
"""
from __future__ import annotations
import sys
import time
import json
import os

import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/archscope/src")

from archscope import circuits
from archscope.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"
PYTHIA_NAME = "EleutherAI/pythia-160m"
MAMBA_NAME = "state-spaces/mamba-130m-hf"


def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[setup] Loading 3 models…")
    pythia_tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if pythia_tok.pad_token is None: pythia_tok.pad_token = pythia_tok.eos_token
    pythia = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, dtype=torch.float32); pythia.eval()

    mamba_tok = AutoTokenizer.from_pretrained(MAMBA_NAME)
    if mamba_tok.pad_token is None: mamba_tok.pad_token = mamba_tok.eos_token
    mamba = AutoModelForCausalLM.from_pretrained(MAMBA_NAME, dtype=torch.float32); mamba.eval()

    kazdov, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)

    models = [
        ("Pythia-160m  (transformer)",       pythia, pythia_tok),
        ("kazdov-α      (hybrid MoBE-BCN)",   kazdov, kazdov_tok),
        ("Mamba-130m   (true SSM)",          mamba, mamba_tok),
    ]

    print()
    print("="*92)
    print("CIRCUIT DETECTION MATRIX — induction head, copy, early-token concentration")
    print("="*92)
    print(f"\n{'Model':<35} | {'induction':>9} | {'copy':>6} | {'concentration':>13}")
    print(f"{'-'*35}-+-{'-'*9}-+-{'-'*6}-+-{'-'*13}")

    all_results = {}
    for name, model, tok in models:
        print(f"\n  Running on {name}…")
        try:
            scores = circuits.run_all_circuits(model, tokenizer=tok, device="cpu")
            ind = scores.get("induction_head")
            cpy = scores.get("copy_circuit")
            ent = scores.get("early_token_concentration")

            ind_str = f"{ind.relative:>5.1f}x" if ind else "  —  "
            cpy_str = f"{cpy.score:>5.2%}" if cpy else "  —  "
            ent_str = f"{ent.relative:>5.2f}" if ent else "  —  "
            all_results[name] = {
                k: {
                    "score": v.score, "baseline": v.baseline, "relative": v.relative,
                    "raw": v.raw,
                } for k, v in scores.items()
            }
            print(f"  {name:<35} | {ind_str:>9} | {cpy_str:>6} | {ent_str:>13}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Pretty re-print summary
    print()
    print("="*92)
    print("SUMMARY TABLE")
    print("="*92)
    print(f"\n{'Model':<35} | {'induction':>10} | {'copy':>8} | {'concentration':>14}")
    print(f"{'(rel. to baseline)':<35} | {'(× chance)':>10} | {'(% acc)':>8} | {'(rel. to max)':>14}")
    print(f"{'-'*35}-+-{'-'*10}-+-{'-'*8}-+-{'-'*14}")
    for name in all_results:
        r = all_results[name]
        ind = r.get("induction_head", {})
        cpy = r.get("copy_circuit", {})
        ent = r.get("early_token_concentration", {})
        print(f"{name:<35} | "
              f"{ind.get('relative', float('nan')):>10.1f} | "
              f"{cpy.get('score', float('nan')):>8.2%} | "
              f"{ent.get('relative', float('nan')):>14.3f}")

    print("\nINTERPRETATION:")
    print("  • induction relative ≫ 1   → induction head present (model copies A→B)")
    print("  • copy accuracy > 30%       → strong copy circuit")
    print("  • concentration relative ≈ 1 → uniform predictions (no concentration)")
    print("  • concentration relative ≈ 0 → highly confident predictions (concentrated)")

    # Save
    out_path = "/Users/kazdov/code/OriginalKazdov/archscope/_research/circuits_3arch.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {out_path}")
    print(f"\nRuntime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
