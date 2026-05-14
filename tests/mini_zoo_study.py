"""Mini-zoo study: run InterpBench on 7 diverse small models.

Generates the comparative table that's the centerpiece of the blog post.

Models tested (each <600M params, runnable on CPU):
- EleutherAI/pythia-160m         — transformer baseline
- EleutherAI/pythia-410m         — same arch, 2.5x scale
- gpt2                            — different transformer family
- state-spaces/mamba-130m-hf     — SSM
- state-spaces/mamba-370m-hf     — SSM scale (2.8x)
- Qwen/Qwen2.5-0.5B               — modern transformer
- kazdov-α-98m                    — hybrid MoBE-BCN+MHA (our own)
"""
from __future__ import annotations
import sys
import os
import time
import json
import traceback
from dataclasses import asdict

import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/polylens/src")

from polylens import bench
from polylens.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"

# (model_name, hf_id, backend_hint, arch_family, extra_kwargs)
ZOO = [
    ("Pythia-160m",  "EleutherAI/pythia-160m",         "transformer", "transformer", {}),
    ("Pythia-410m",  "EleutherAI/pythia-410m",         "transformer", "transformer", {}),
    ("GPT-2",        "gpt2",                            "transformer", "transformer", {}),
    ("Mamba-130m",   "state-spaces/mamba-130m-hf",     "mamba",       "ssm",         {"ssm_layer": 12}),
    ("Mamba-370m",   "state-spaces/mamba-370m-hf",     "mamba",       "ssm",         {"ssm_layer": 24}),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",              "transformer", "transformer", {}),
    # Kazdov is handled specially below
]


def make_hf_tokenize(tok):
    def fn(texts):
        return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return fn


def make_kazdov_tokenize(tok):
    def fn(texts):
        out = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}
    return fn


def run_one(display_name, hf_id, backend_hint, arch_family, extra):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[loading] {display_name} ({hf_id})…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; running InterpBench…")
    t1 = time.time()
    profile = bench.benchmark(
        model_name=hf_id, model=model, tokenizer=tok,
        backend_hint=backend_hint, arch_family=arch_family,
        tokenize_fn=make_hf_tokenize(tok),
        **extra,
    )
    print(f"  bench in {time.time()-t1:.1f}s. induction={profile.induction_head_relative:.0f}×, "
          f"sae_dense={profile.sae_dense_recon:.4f}, sae_r1={profile.sae_rank1_recon:.4f}")
    # Free model memory
    del model
    return profile


def run_kazdov():
    print(f"\n[loading] kazdov-α-98m (local)…")
    model, tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)
    profile = bench.benchmark(
        model_name="kazdov-α-98m", model=model, tokenizer=tok,
        backend_hint="kazdov", arch_family="hybrid",
        tokenize_fn=make_kazdov_tokenize(tok),
    )
    print(f"  bench done. induction={profile.induction_head_relative:.0f}×")
    return profile


def main():
    t_start = time.time()
    profiles = []
    notes = []

    for entry in ZOO:
        display, hf_id, hint, fam, extra = entry
        try:
            p = run_one(display, hf_id, hint, fam, extra)
            p.model_name = display   # short name for table
            profiles.append(p)
        except Exception as e:
            print(f"  [SKIPPED] {display}: {type(e).__name__}: {str(e)[:80]}")
            notes.append(f"{display}: skipped — {type(e).__name__}")
            traceback.print_exc(limit=2)

    try:
        p = run_kazdov()
        profiles.append(p)
    except Exception as e:
        print(f"  [SKIPPED] kazdov: {e}")
        notes.append(f"kazdov: skipped — {e}")

    # Final summary
    print()
    print("="*102)
    print("MINI-ZOO LEADERBOARD")
    print("="*102)
    cols = ("model", "arch", "params", "L", "d", "sent", "math", "induct", "copy", "conc", "saeD", "saeR1", "ssmVar")
    fmt = "{:<14} {:<10} {:>6} {:>3} {:>4} {:>5} {:>5} {:>8} {:>5} {:>5} {:>6} {:>6} {:>6}"
    print(fmt.format(*cols))
    print("-"*102)
    for p in profiles:
        params_str = f"{p.n_params/1e6:.0f}M"
        print(fmt.format(
            p.model_name[:14], p.arch_family[:10],
            params_str, p.n_layers, p.hidden_dim,
            f"{p.probe_sentiment_auroc:.2f}",
            f"{p.probe_math_auroc:.2f}",
            f"{p.induction_head_relative:.0f}",
            f"{p.copy_accuracy:.0%}",
            f"{p.concentration_relative:.2f}",
            f"{p.sae_dense_recon:.3f}",
            f"{p.sae_rank1_recon:.3f}",
            f"{p.ssm_state_variance_ratio:.2f}" if p.ssm_state_variance_ratio == p.ssm_state_variance_ratio else "—",
        ))

    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")

    out_path = "/Users/kazdov/code/OriginalKazdov/polylens/_research/mini_zoo_leaderboard.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(p) for p in profiles], f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
