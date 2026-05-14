"""mechinterp-small quickstart — 4 demos in one file.

Demo 1: Probe sentiment in Pythia
Demo 2: Detect induction circuits across 2 archs
Demo 3: Train SAE on Mamba SSM state (unique to this library)
Demo 4: Compare InterpBench profile for 2 models

Run: python examples/quickstart.py
"""
import torch
import mechinterp_small as mi
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    return model, tok


def tokenize_fn(tok):
    return lambda texts: tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


# === Demo 1: probe sentiment ===
print("\n=== Demo 1: Probe sentiment in Pythia ===")
model, tok = load("EleutherAI/pythia-160m")
tk = tokenize_fn(tok)
pos = ["I love this", "Amazing show", "Wonderful day"]
neg = ["I hate this", "Awful film", "Terrible day"]
probe = mi.probes.fit_probe(
    model, inputs_pos=tk(pos), inputs_neg=tk(neg),
    layer_name="layer_5.residual", backend_hint="transformer",
)
print(f"  train_auroc: {probe.metrics['train_auroc']:.3f}")


# === Demo 2: induction circuit detection ===
print("\n=== Demo 2: Induction head detection ===")
scores = mi.circuits.induction_head_score(model, n_trials=30)
print(f"  Pythia induction relative to chance: {scores.relative:.1f}×")


# === Demo 3: Mamba SSM state SAE (unique) ===
print("\n=== Demo 3: Train SAE on Mamba SSM state ===")
mamba_model, mamba_tok = load("state-spaces/mamba-130m-hf")
mtk = tokenize_fn(mamba_tok)
backend = mi.backends.Backend.for_model(mamba_model, hint="mamba")
inputs = mtk(["The cat sat on the mat.", "Solve x squared equals 16."])
rec = backend.extract(inputs, layers=["layer_5.ssm_state"])[0]
print(f"  SSM state shape: {tuple(rec.activations.shape)}  (B × d_inner × d_state)")
acts = rec.activations.reshape(rec.activations.shape[0], -1).detach()
cfg = mi.sae.SAEConfig(input_dim=acts.shape[-1], n_features=256, sae_type="dense",
                       sparsity=1e-4, learning_rate=3e-3)
trained = mi.sae.fit_sae(acts, cfg, epochs=30)
print(f"  SAE trained on SSM state: recon={trained.last_metrics['recon']:.4f}")


# === Demo 4: InterpBench profile comparison ===
print("\n=== Demo 4: InterpBench compact comparison ===")
print("  (skip full run — see tests/run_interpbench_leaderboard.py)")
print("  Or: `mechinterp bench EleutherAI/pythia-160m --arch transformer`")

print("\nDone.")
