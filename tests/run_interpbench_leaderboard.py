"""Run full InterpBench across our 3 reference models to seed the leaderboard."""
from __future__ import annotations
import sys
import os
import json
import time
import torch
from dataclasses import asdict

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/mechinterp-small/src")

from mechinterp_small import bench
from mechinterp_small.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"


def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    profiles = []

    # ---- Pythia ----
    print("[1/3] Pythia-160m…")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", dtype=torch.float32)
    model.eval()
    def tk_hf(texts, t=tok): return t(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    p = bench.benchmark(
        "EleutherAI/pythia-160m", model, tok, backend_hint="transformer",
        arch_family="transformer", tokenize_fn=tk_hf,
    )
    profiles.append(p)

    # ---- Mamba ----
    print("[2/3] Mamba-130m-hf…")
    tok2 = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    if tok2.pad_token is None: tok2.pad_token = tok2.eos_token
    model2 = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float32)
    model2.eval()
    def tk_hf2(texts, t=tok2): return t(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    p2 = bench.benchmark(
        "state-spaces/mamba-130m-hf", model2, tok2, backend_hint="mamba",
        arch_family="ssm", tokenize_fn=tk_hf2, ssm_layer=12,
    )
    profiles.append(p2)

    # ---- Kazdov ----
    print("[3/3] Kazdov-α 98m…")
    kazdov_model, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)
    def tk_kazdov(texts, t=kazdov_tok):
        out = t(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}
    p3 = bench.benchmark(
        "kazdov-α-98m", kazdov_model, kazdov_tok, backend_hint="kazdov",
        arch_family="hybrid", tokenize_fn=tk_kazdov,
    )
    profiles.append(p3)

    # ---- Print markdown leaderboard ----
    print()
    print("="*92)
    print("INTERPBENCH LEADERBOARD v0.1")
    print("="*92)
    for p in profiles:
        print()
        print(bench.profile_to_markdown(p))

    # ---- Compact comparison table ----
    print()
    print("="*92)
    print("COMPACT COMPARISON TABLE")
    print("="*92)
    headers = ["model", "arch", "sent", "math", "induct(×)", "copy%", "conc", "saeD", "saeR1", "ssmVar"]
    print(f"\n{'model':<28} {'arch':<10} {'sent':>5} {'math':>5} {'induct':>9} {'copy':>5} {'conc':>5} {'saeD':>6} {'saeR1':>6} {'ssmVar':>6}")
    print("-"*92)
    for p in profiles:
        print(f"{p.model_name[:28]:<28} {p.arch_family:<10} "
              f"{p.probe_sentiment_auroc:>5.2f} {p.probe_math_auroc:>5.2f} "
              f"{p.induction_head_relative:>9.0f} {p.copy_accuracy:>5.1%} "
              f"{p.concentration_relative:>5.2f} {p.sae_dense_recon:>6.3f} "
              f"{p.sae_rank1_recon:>6.3f} {p.ssm_state_variance_ratio:>6.3f}")

    # ---- Save JSON leaderboard ----
    out_path = "/Users/kazdov/code/OriginalKazdov/mechinterp-small/_research/interpbench_leaderboard_v01.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(p) for p in profiles], f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
