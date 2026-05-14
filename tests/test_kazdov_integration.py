"""End-to-end test: apply archscope to kazdov-α (hybrid attention).

Validates that the same 4 methods that worked on Pythia also work on
kazdov-α's hybrid MoBE-BCN+MHA architecture. This is the core cross-arch
test for the workshop paper.
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/archscope/src")

from archscope import probes, sae, neurons
from archscope.backends import Backend
from archscope.kazdov_backend import load_kazdov_checkpoint


CHECKPOINT = "/Users/kazdov/code/OriginalKazdov/_models/kazdov-98m-alpha"


def tokenize(tokenizer, texts: list[str]) -> dict:
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"].bool()}


def main():
    t_start = time.time()
    print(f"[setup] Loading kazdov-α from {CHECKPOINT}…")
    model, tokenizer = load_kazdov_checkpoint(CHECKPOINT, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] Loaded. n_layers={len(model.blocks)}, d_model={model.d_model}, n_params={n_params/1e6:.1f}M")

    # ----- Test 1: Backend extract -----
    print("\n[test 1] KazdovBackend.extract — pull residual stream from blocks")
    backend = Backend.for_model(model, hint="kazdov")
    inputs = tokenize(tokenizer, ["The integral of x squared is", "Solve for x: 2x + 3 = 7"])
    records = backend.extract(inputs, layers=["layer_5.residual"])
    rec = records[0]
    print(f"  layer: {rec.layer_name}, shape={tuple(rec.activations.shape)}, meta={rec.meta}")
    assert rec.activations.dim() == 3 and rec.activations.shape[-1] == model.d_model
    print("  ✓ PASS")

    # ----- Test 2: Probe fit on math vs non-math -----
    print("\n[test 2] Probe fit — math vs non-math prompts via layer 5 residual")
    math_texts = [
        "Solve for x: 2x + 3 = 11",
        "Compute the derivative of x^3",
        "The integral from 0 to 1 of x dx is",
        "If a triangle has angles 30 60 90 degrees",
        "The eigenvalue of matrix M is",
        "By the chain rule, d/dx of sin(x^2) is",
        "Find the roots of x^2 - 5x + 6",
        "The Taylor series of e^x at zero is",
    ]
    nonmath_texts = [
        "The cat sat on the mat softly.",
        "Music has the power to move us.",
        "Mountains stretch to the horizon.",
        "She whispered her secret carefully.",
        "Birds sing at dawn every day.",
        "The chef prepared dinner slowly.",
        "Children laughed in the park.",
        "Rain pattered against the window.",
    ]
    inputs_m = tokenize(tokenizer, math_texts)
    inputs_n = tokenize(tokenizer, nonmath_texts)

    pf = probes.fit_probe(
        model,
        inputs_pos=inputs_m,
        inputs_neg=inputs_n,
        layer_name="layer_5.residual",
        backend_hint="kazdov",
    )
    print(f"  train_auroc: {pf.metrics['train_auroc']:.3f}")
    print(f"  val_auroc:   {pf.metrics['val_auroc']:.3f}")
    print(f"  train_loss:  {pf.metrics['train_loss']:.4f}")
    # Kazdov-α is math-trained — math vs non-math should be very separable
    assert pf.metrics["train_auroc"] > 0.6, f"Probe failed on kazdov: auroc={pf.metrics['train_auroc']}"
    print("  ✓ PASS (probe learns on math-vs-non-math)")

    # ----- Test 3: SAE on real kazdov activations -----
    print("\n[test 3] Dense SAE on kazdov residual stream")
    diverse_texts = math_texts + nonmath_texts + [
        "Prove that the sum of two evens is even.",
        "The Cauchy-Schwarz inequality states",
        "Let f be a continuous function on [0,1]",
        "Define the limit as n approaches infinity",
        "By induction on the natural numbers",
        "The dot product of u and v is",
        "A group is abelian if and only if",
        "The fundamental theorem of calculus says",
    ]
    inputs_div = tokenize(tokenizer, diverse_texts)
    rec_div = backend.extract(inputs_div, layers=["layer_5.residual"])[0]
    acts = rec_div.activations.reshape(-1, model.d_model).detach()
    print(f"  training on {len(acts)} vectors (d={model.d_model})")
    cfg = sae.SAEConfig(input_dim=model.d_model, n_features=512, sae_type="dense",
                         sparsity=1e-4, learning_rate=3e-3)
    trained = sae.fit_sae(acts, cfg, epochs=100)
    m = trained.last_metrics
    print(f"  recon: {m['recon']:.4f}, l1: {m['l1']:.4f}, n_active: {m['n_active']:.3f}")
    assert torch.isfinite(torch.tensor(m["recon"]))
    print("  ✓ PASS")

    # ----- Test 4: Rank-1 SAE (WriteSAE-style) -----
    print("\n[test 4] Rank-1 SAE (designed for recurrent cache — testing on kazdov hybrid)")
    cfg2 = sae.SAEConfig(input_dim=model.d_model, n_features=512, sae_type="rank1",
                          sparsity=1e-4, learning_rate=3e-3)
    trained2 = sae.fit_sae(acts, cfg2, epochs=100)
    m2 = trained2.last_metrics
    print(f"  recon: {m2['recon']:.4f}, l1: {m2['l1']:.4f}, n_active: {m2['n_active']:.3f}")
    assert torch.isfinite(torch.tensor(m2["recon"]))
    print("  ✓ PASS")

    # ----- Test 5: Neuron discovery on kazdov -----
    print("\n[test 5] Neuron discovery on kazdov (math vs non-math)")
    edit = neurons.find_neurons(
        model,
        inputs_harmful=inputs_m,    # "harmful" → here treated as math
        inputs_benign=inputs_n,
        config=neurons.NeuronEditConfig(top_frac=0.001),
        backend_hint="kazdov",
    )
    total = sum(len(idx) for idx in edit.layer_to_indices.values())
    sample_layer = list(edit.layer_to_indices.keys())[0]
    print(f"  n_layers: {len(edit.layer_to_indices)}, total neurons: {total}")
    print(f"  sample: {sample_layer} -> {edit.layer_to_indices[sample_layer][:3].tolist()}")
    assert total > 0
    print("  ✓ PASS")

    # ----- Test 6: Apply neuron edit on kazdov via hook -----
    print("\n[test 6] Neuron edit apply on kazdov — verify hook propagates")
    test_input = tokenize(tokenizer, ["Compute 2 plus 2 equals"])
    with torch.no_grad():
        base_logits = model(test_input["input_ids"]).get("logits") if isinstance(model(test_input["input_ids"]), dict) else model(test_input["input_ids"])
        if isinstance(base_logits, dict):
            base_logits = base_logits["logits"]
        base_last = base_logits[:, -1, :].clone()
    edit.multiplier = 0.0  # ablate
    with edit.apply_hook(model):
        with torch.no_grad():
            mod_logits = model(test_input["input_ids"])
            if isinstance(mod_logits, dict):
                mod_logits = mod_logits["logits"]
            mod_last = mod_logits[:, -1, :].clone()
    diff = (base_last - mod_last).abs().mean().item()
    print(f"  mean |Δ logits|: {diff:.6f}")
    assert diff > 1e-6, "Hook didn't propagate to kazdov output"
    print("  ✓ PASS (hooks work on kazdov.blocks[i])")

    print(f"\n{'='*60}\nCROSS-ARCH SUCCESS: 6/6 on kazdov-α (hybrid MoBE-BCN+MHA)")
    print(f"Total runtime: {time.time()-t_start:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
