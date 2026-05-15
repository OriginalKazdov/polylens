"""Cross-Architecture Mechanistic Interpretability — first comparative table.

For 3 binary tasks (sentiment, math-vs-nonmath, length-bucket), we ask:
- Does a linear probe at layer L distinguish the two classes?
- How does probe AUROC vary by LAYER and by ARCHITECTURE?

Both models receive IDENTICAL text inputs.
Pythia-160m (12 layers, d=768, standard transformer)
kazdov-α (12 layers, d=512, MoBE-BCN+MHA hybrid, math-trained)

This generates the first concrete table for the workshop paper.
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from archscope import probes
from archscope.backends import Backend
import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "scripts"))
from _kazdov_loader import load_kazdov_checkpoint


CHECKPOINT_KAZDOV = __import__("os").environ.get("KAZDOV_CHECKPOINT", "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha")
PYTHIA_NAME = "EleutherAI/pythia-160m"


# ---------- DATASETS (binary classification tasks) ----------

SENTIMENT_POS = [
    "I love this movie, it's amazing!", "What a wonderful day.", "Best book I have read.",
    "She is so kind and thoughtful.", "The food was delicious.", "I'm thrilled about the news.",
    "Such a beautiful sunset tonight.", "He always makes me laugh.", "Amazing performance overall.",
    "Truly a delightful experience.", "Incredible work, congratulations!", "I admire her dedication.",
    "The concert was breathtaking.", "He is the kindest person I know.", "This was pure joy.",
    "I had a wonderful evening.",
]
SENTIMENT_NEG = [
    "I hate this place.", "What a terrible movie.", "This is the worst day ever.",
    "She is mean and selfish.", "The food was awful.", "I'm so disappointed.",
    "Such a horrible meeting.", "He always annoys me.", "What an awful experience.",
    "I despise everything about this.", "This restaurant is dreadful.", "She is the rudest person.",
    "The concert was a disaster.", "He never listens to anyone.", "Truly a miserable evening.",
    "I regret coming here.",
]

MATH = [
    "Solve for x: 2x + 3 = 11", "Compute the derivative of x^3",
    "The integral from 0 to 1 of x dx is", "Triangle angles 30, 60, 90",
    "The eigenvalue of matrix M is", "By chain rule, d/dx of sin(x^2)",
    "Find roots of x^2 - 5x + 6", "The Taylor series of e^x at 0",
    "Prove the sum of two evens is even.", "The Cauchy-Schwarz inequality states",
    "Let f be a continuous function on [0,1].", "Define the limit as n approaches infinity",
    "By induction on the natural numbers", "The dot product of u and v",
    "A group is abelian if and only if", "The fundamental theorem of calculus",
]
NONMATH = [
    "The cat sat on the mat softly.", "Music has the power to move us.",
    "Mountains stretch to the horizon.", "She whispered her secret carefully.",
    "Birds sing at dawn every day.", "The chef prepared dinner slowly.",
    "Children laughed in the park.", "Rain pattered against the window.",
    "Coffee aroma filled the kitchen.", "Time passes faster when busy.",
    "Stars appeared in the dark sky.", "The river flowed gently downstream.",
    "Snow blanketed the entire valley.", "Dance is a universal language.",
    "Books contain endless worlds.", "Travel broadens one's perspective.",
]


def tokenize_hf(tokenizer, texts):
    """For HF transformers (Pythia)."""
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


def tokenize_kazdov(tokenizer, texts):
    """For kazdov (returns input_ids + attention_mask as bool)."""
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}


# ---------- PROBE SWEEP across all layers ----------

def probe_per_layer(model, backend_hint, tokenizer_fn, pos_texts, neg_texts, task_name):
    """Train a probe at each layer; return AUROC vs layer index."""
    backend = Backend.for_model(model, hint=backend_hint)
    n_layers = len(backend.layer_names())
    inputs_pos = tokenizer_fn(pos_texts)
    inputs_neg = tokenizer_fn(neg_texts)

    results = []
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx}.residual"
        try:
            pf = probes.fit_probe(
                model,
                inputs_pos=inputs_pos,
                inputs_neg=inputs_neg,
                layer_name=layer_name,
                backend_hint=backend_hint,
            )
            results.append({
                "layer": layer_idx,
                "train_auroc": pf.metrics["train_auroc"],
                "val_auroc": pf.metrics["val_auroc"],
                "loss": pf.metrics["train_loss"],
            })
        except Exception as e:
            results.append({"layer": layer_idx, "error": str(e)[:80]})

    return {"task": task_name, "results": results}


def print_layer_table(arch_name, task_results: list):
    print(f"\n  {arch_name}")
    print(f"  {'Layer':>5} | {'train_auroc':>11} | {'val_auroc':>9} | {'loss':>6}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*9}-+-{'-'*6}")
    for r in task_results:
        if "error" in r:
            print(f"  {r['layer']:>5} | ERROR: {r['error']}")
        else:
            print(f"  {r['layer']:>5} | {r['train_auroc']:>11.3f} | {r['val_auroc']:>9.3f} | {r['loss']:>6.3f}")


# ---------- MAIN ----------

def main():
    t_start = time.time()

    # Load both models
    print(f"[setup] Loading Pythia-160m…")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pythia_tok = AutoTokenizer.from_pretrained(PYTHIA_NAME)
    if pythia_tok.pad_token is None: pythia_tok.pad_token = pythia_tok.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained(PYTHIA_NAME, dtype=torch.float32)
    pythia_model.eval()
    print(f"  Pythia: {pythia_model.config.num_hidden_layers} layers, d={pythia_model.config.hidden_size}")

    print(f"[setup] Loading kazdov-α…")
    kazdov_model, kazdov_tok = load_kazdov_checkpoint(CHECKPOINT_KAZDOV)
    print(f"  Kazdov: {len(kazdov_model.blocks)} layers, d={kazdov_model.d_model}")
    print()

    tasks = [
        ("Sentiment (pos vs neg)", SENTIMENT_POS, SENTIMENT_NEG),
        ("Math vs Non-math",       MATH,           NONMATH),
    ]

    print(f"{'='*72}")
    print(f"CROSS-ARCHITECTURE PROBE COMPARISON — layer-wise AUROC")
    print(f"{'='*72}")

    all_results = {}
    for task_name, pos, neg in tasks:
        print(f"\n### TASK: {task_name}")

        py_results = probe_per_layer(
            pythia_model, "transformer",
            lambda t: tokenize_hf(pythia_tok, t),
            pos, neg, task_name
        )
        kz_results = probe_per_layer(
            kazdov_model, "kazdov",
            lambda t: tokenize_kazdov(kazdov_tok, t),
            pos, neg, task_name
        )
        print_layer_table("Pythia-160m (transformer, general-pretrain)", py_results["results"])
        print_layer_table("kazdov-α (MoBE-BCN+MHA, math-pretrain)", kz_results["results"])

        # Aggregate stats
        py_aurocs = [r["val_auroc"] for r in py_results["results"] if "val_auroc" in r and not (r["val_auroc"] != r["val_auroc"])]
        kz_aurocs = [r["val_auroc"] for r in kz_results["results"] if "val_auroc" in r and not (r["val_auroc"] != r["val_auroc"])]
        if py_aurocs and kz_aurocs:
            py_max_layer = max(range(len(py_results["results"])), key=lambda i: py_results["results"][i].get("val_auroc") or 0)
            kz_max_layer = max(range(len(kz_results["results"])), key=lambda i: kz_results["results"][i].get("val_auroc") or 0)
            print(f"\n  Pythia peak: layer {py_max_layer} (val_auroc={py_results['results'][py_max_layer].get('val_auroc'):.3f})")
            print(f"  Kazdov peak: layer {kz_max_layer} (val_auroc={kz_results['results'][kz_max_layer].get('val_auroc'):.3f})")

        all_results[task_name] = {"pythia": py_results, "kazdov": kz_results}

    print(f"\n{'='*72}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")
    print(f"{'='*72}")

    # Save results JSON for later analysis
    import json
    out_path = str(__import__("pathlib").Path(__file__).parent.parent / "_research" / "cross_arch_results.json")
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
