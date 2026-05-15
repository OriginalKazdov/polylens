"""Experiment v2 — harder dataset, larger test set, cleaner write-up.

Same hypothesis: do sentiment probes transfer Pythia ↔ Mamba?

Changes from v1:
- ~60 pos + ~60 neg examples, mix of explicit and subtler cases
- Activation extraction cached per (model, layer) — no re-extraction in transfer
- Skip the full layer×layer transfer matrix; do per-layer trajectory + best-to-best transfer
- Run 3 seeds for the per-layer probes to get error bars
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

import archscope as ai
from archscope.transfer import learn_alignment
from archscope.probes import ProbeFit, ProbeConfig, _auroc


OUT = Path("/tmp/sentiment_transfer_v2_results")
OUT.mkdir(exist_ok=True)


# ---------- Dataset: bigger and harder ----------

POS = [
    # Explicit (easy)
    "I love this movie", "Wonderful experience", "Amazing work",
    "Fantastic, I'm so happy", "What a beautiful day", "Excellent service",
    "She was absolutely brilliant", "This is the best one yet",
    "I'm thrilled with the result", "Such a delightful read",
    "Truly inspiring story", "Perfect, exactly what I wanted",
    "Marvelous performance", "He did a wonderful job",
    "Outstanding effort by everyone", "Just delightful",
    "I'm in love with it", "Pure joy from start to end",
    "Magnificent achievement", "Couldn't be happier",
    # Subtler positive (harder)
    "It exceeded what I expected", "Worth every penny",
    "I'd buy this again in a heartbeat", "Pleasantly surprised",
    "More than I hoped for", "Solid, dependable, gets the job done",
    "A breath of fresh air", "I find myself smiling thinking about it",
    "Beyond satisfied", "Definitely worth the wait",
    # Slightly mixed but net positive
    "Not bad at all, actually quite good", "Better than I feared",
    "It's not perfect but I'll take it", "Slow start but it really delivers",
    "Has flaws but the good outweighs the bad",
    # Implicit recommendation
    "I've already told three friends about it", "I'll be coming back",
    "Bought another one for my mother", "Worth recommending",
    "Going to be hard to top this",
    # General praise
    "Brilliant decision", "Couldn't ask for more",
    "Top tier work", "Genuinely impressive",
    "Five stars from me",
]
NEG = [
    # Explicit (easy)
    "I hate this movie", "Terrible experience", "Awful work",
    "Disappointing, I'm so upset", "What a miserable day", "Horrible service",
    "She was completely incompetent", "This is the worst one yet",
    "I'm furious with the result", "Such a tedious read",
    "Painfully boring story", "Useless, not what I wanted",
    "Disastrous performance", "He did a horrible job",
    "Pathetic effort by everyone", "Just dreadful",
    "I'm sick of it", "Pure misery from start to end",
    "Catastrophic failure", "Couldn't be more annoyed",
    # Subtler negative (harder)
    "It fell short of what I expected", "Not worth the price",
    "I would not buy this again", "Unpleasantly surprised",
    "Less than I hoped for", "Sloppy, unreliable, falls short",
    "Stale, predictable, tiresome", "I shudder to think about it",
    "Beyond disappointed", "Definitely not worth the wait",
    # Slightly mixed but net negative
    "Not good at all, actually quite poor", "Worse than I feared",
    "It has its moments but no", "Promising start but it really fails",
    "Has merits but the bad outweighs the good",
    # Implicit non-recommendation
    "I've already warned three friends about it", "I won't be coming back",
    "Returned it the next day", "Would not recommend",
    "Going to be hard to find something worse",
    # General critique
    "Poor decision", "Couldn't get past the issues",
    "Bottom tier work", "Genuinely unimpressive",
    "One star from me",
]
print(f"Dataset: {len(POS)} pos + {len(NEG)} neg = {len(POS)+len(NEG)} total")

# 80/20 stratified split. Same seed → same split for both archs.
import random
rng = random.Random(0)
def split(items, frac):
    items = list(items)
    rng.shuffle(items)
    k = int(len(items) * frac)
    return items[:k], items[k:]
TRAIN_POS, TEST_POS = split(POS, 0.8)
TRAIN_NEG, TEST_NEG = split(NEG, 0.8)
print(f"Train: {len(TRAIN_POS)} pos + {len(TRAIN_NEG)} neg = {len(TRAIN_POS)+len(TRAIN_NEG)}")
print(f"Test:  {len(TEST_POS)} pos + {len(TEST_NEG)} neg = {len(TEST_POS)+len(TEST_NEG)}\n")


# ---------- Helpers ----------

def tokenize_batch(tokenizer, texts):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=24)


def extract_last_token_per_layer(backend, tokenizer, texts):
    """Extract activations for all layers in ONE forward pass, at each row's
    real last-token position. Returns: {layer_name: tensor(B, H)}.
    """
    inp = tokenize_batch(tokenizer, texts)
    inputs = {"input_ids": inp["input_ids"]}
    if "attention_mask" in inp:
        inputs["attention_mask"] = inp["attention_mask"]
        lens = inp["attention_mask"].sum(dim=1)
        idx = (lens - 1).clamp(min=0)
    else:
        idx = None
    residual_layers = [n for n in backend.layer_names() if ".residual" in n]
    recs = backend.extract(inputs, layers=residual_layers)
    arange = torch.arange(inputs["input_ids"].shape[0])
    if idx is None:
        idx = torch.full((inputs["input_ids"].shape[0],),
                         inputs["input_ids"].shape[1] - 1, dtype=torch.long)
    return {rec.layer_name: rec.activations[arange, idx] for rec in recs}


def train_probe(acts_train, labels_train, acts_test, labels_test, n_seeds=3):
    """Return (mean_test_auroc, std_test_auroc, direction_seed0, bias_seed0)."""
    H = acts_train.shape[-1]
    aurocs = []
    first_dir, first_bias = None, None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        cfg = ProbeConfig(layer_name="x", probe_type="linear")
        pf = ProbeFit(cfg, input_dim=H)
        pf.train(acts_train, labels_train, epochs=100, lr=5e-3, batch_size=32)
        with torch.no_grad():
            logits = pf.probe(acts_test)
        aurocs.append(float(_auroc(logits, labels_test)))
        if seed == 0:
            # Clean accessors added in v0.2.7
            first_dir, first_bias = pf.direction, pf.bias
    return float(sum(aurocs) / n_seeds), float(torch.tensor(aurocs).std().item()), first_dir, first_bias


# ---------- Load both models ----------

t = time.time()
pythia_model, pythia_tok, pythia_backend = ai.load_model(
    "EleutherAI/pythia-160m", arch="transformer")
print(f"[load] Pythia-160m: {time.time()-t:.1f}s")

t = time.time()
mamba_model, mamba_tok, mamba_backend = ai.load_model(
    "state-spaces/mamba-130m-hf", arch="mamba")
print(f"[load] Mamba-130m: {time.time()-t:.1f}s\n")


# ---------- One-shot extraction per arch (every layer, train + test) ----------

print("[extract] Pythia train+test through all layers in 2 forward passes...")
t = time.time()
ALL_TRAIN = TRAIN_POS + TRAIN_NEG
ALL_TEST  = TEST_POS  + TEST_NEG
pythia_train = extract_last_token_per_layer(pythia_backend, pythia_tok, ALL_TRAIN)
pythia_test  = extract_last_token_per_layer(pythia_backend, pythia_tok, ALL_TEST)
print(f"  {time.time()-t:.1f}s")

print("[extract] Mamba train+test through all layers in 2 forward passes...")
t = time.time()
mamba_train = extract_last_token_per_layer(mamba_backend, mamba_tok, ALL_TRAIN)
mamba_test  = extract_last_token_per_layer(mamba_backend, mamba_tok, ALL_TEST)
print(f"  {time.time()-t:.1f}s\n")

labels_train = torch.cat([torch.ones(len(TRAIN_POS)), torch.zeros(len(TRAIN_NEG))])
labels_test  = torch.cat([torch.ones(len(TEST_POS)),  torch.zeros(len(TEST_NEG))])


# ---------- Per-layer probes (3 seeds each) ----------

def per_layer_probes(train_dict, test_dict, name):
    out = {}
    print(f"[probes] {name} — 3-seed probe per layer:")
    for ln in train_dict:
        m, s, d, b = train_probe(train_dict[ln], labels_train,
                                  test_dict[ln],  labels_test, n_seeds=3)
        out[ln] = {"test_auroc_mean": m, "test_auroc_std": s,
                   "direction": d, "bias": b}
        print(f"  {name:8s} {ln:24s}  AUROC = {m:.3f} ± {s:.3f}")
    return out

pythia_probes = per_layer_probes(pythia_train, pythia_test, "Pythia")
print()
mamba_probes  = per_layer_probes(mamba_train,  mamba_test,  "Mamba")


# ---------- Best layer per model ----------

best_pythia_ln = max(pythia_probes, key=lambda k: pythia_probes[k]["test_auroc_mean"])
best_mamba_ln  = max(mamba_probes,  key=lambda k: mamba_probes[k]["test_auroc_mean"])
print(f"\nBest Pythia layer: {best_pythia_ln}  AUROC={pythia_probes[best_pythia_ln]['test_auroc_mean']:.3f}")
print(f"Best Mamba layer:  {best_mamba_ln}   AUROC={mamba_probes[best_mamba_ln]['test_auroc_mean']:.3f}")


# ---------- Best-to-best cross-arch transfer ----------

def transfer_probe(src_dir, src_bias, src_paired, tgt_paired, tgt_test, labels_test):
    """Learn M: tgt -> src (ridge regression). Map tgt_test → src space.
    Apply src probe direction + bias. Return test AUROC.
    """
    M = learn_alignment(src_paired, tgt_paired, ridge=0.001)   # M: tgt → src
    tgt_test_in_src = tgt_test @ M.T
    logits = tgt_test_in_src @ src_dir + src_bias               # (N,)
    return float(_auroc(logits, labels_test))

print("\n[transfer] best-to-best Pythia ↔ Mamba ...")
p2m = transfer_probe(
    pythia_probes[best_pythia_ln]["direction"],
    pythia_probes[best_pythia_ln]["bias"],
    pythia_train[best_pythia_ln], mamba_train[best_mamba_ln],
    mamba_test[best_mamba_ln], labels_test,
)
print(f"  Pythia probe ({best_pythia_ln}) → Mamba ({best_mamba_ln}): AUROC {p2m:.3f}")

m2p = transfer_probe(
    mamba_probes[best_mamba_ln]["direction"],
    mamba_probes[best_mamba_ln]["bias"],
    mamba_train[best_mamba_ln], pythia_train[best_pythia_ln],
    pythia_test[best_pythia_ln], labels_test,
)
print(f"  Mamba probe  ({best_mamba_ln}) → Pythia ({best_pythia_ln}): AUROC {m2p:.3f}")


# ---------- Save ----------

def serialize_probes(d):
    return {ln: {"test_auroc_mean": v["test_auroc_mean"],
                 "test_auroc_std":  v["test_auroc_std"]}
            for ln, v in d.items()}

results = {
    "dataset": {
        "n_pos": len(POS), "n_neg": len(NEG),
        "n_train": len(TRAIN_POS) + len(TRAIN_NEG),
        "n_test": len(TEST_POS) + len(TEST_NEG),
    },
    "pythia_per_layer": serialize_probes(pythia_probes),
    "mamba_per_layer":  serialize_probes(mamba_probes),
    "best_pythia_layer": best_pythia_ln,
    "best_mamba_layer":  best_mamba_ln,
    "best_pythia_auroc": pythia_probes[best_pythia_ln]["test_auroc_mean"],
    "best_mamba_auroc":  mamba_probes[best_mamba_ln]["test_auroc_mean"],
    "transfer_pythia_to_mamba": p2m,
    "transfer_mamba_to_pythia": m2p,
}
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {OUT / 'results.json'}")

# Markdown
md = [
    "# Sentiment probe transfer: Pythia-160m ↔ Mamba-130m (v2)",
    "",
    "## Setup",
    f"- {len(POS)} positive + {len(NEG)} negative sentiment statements",
    "- 80/20 stratified train/test split (seed=0)",
    "- Mix of explicit ('I love this'), subtle ('exceeded what I expected'), and mixed-net cases",
    "- Linear probe on last-real-token residual stream, 3 seeds per layer",
    "- Cross-arch transfer via `archscope.transfer.learn_alignment` (ridge=0.001)",
    "",
    "## Per-layer probe AUROC",
    "",
    "**Pythia-160m** (12 layers):",
    "",
    "| Layer | Mean AUROC | Std |",
    "|---|---:|---:|",
]
for ln in pythia_probes:
    p = pythia_probes[ln]
    md.append(f"| {ln} | {p['test_auroc_mean']:.3f} | {p['test_auroc_std']:.3f} |")

md += ["", "**Mamba-130m** (24 layers):", "", "| Layer | Mean AUROC | Std |", "|---|---:|---:|"]
for ln in mamba_probes:
    p = mamba_probes[ln]
    md.append(f"| {ln} | {p['test_auroc_mean']:.3f} | {p['test_auroc_std']:.3f} |")

md += [
    "",
    "## Findings",
    "",
    f"- **Best Pythia layer**: {best_pythia_ln} (AUROC {pythia_probes[best_pythia_ln]['test_auroc_mean']:.3f})",
    f"- **Best Mamba layer**:  {best_mamba_ln} (AUROC {mamba_probes[best_mamba_ln]['test_auroc_mean']:.3f})",
    "",
    "## Cross-arch transfer (best → best)",
    "",
    f"- **Pythia probe → Mamba**: AUROC {p2m:.3f} (own-arch best was {mamba_probes[best_mamba_ln]['test_auroc_mean']:.3f})",
    f"- **Mamba probe → Pythia**: AUROC {m2p:.3f} (own-arch best was {pythia_probes[best_pythia_ln]['test_auroc_mean']:.3f})",
    "",
    "Random baseline = 0.500.",
]
with open(OUT / "report.md", "w") as f:
    f.write("\n".join(md))
print(f"Saved markdown to {OUT / 'report.md'}")
print("\nDone.")
