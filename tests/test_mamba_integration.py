"""End-to-end test: apply archscope to Mamba 130m (true SSM).

This is the third arch family for the cross-architecture paper:
- Pythia (standard transformer)         — done
- Kazdov-α (hybrid MoBE-BCN+MHA)        — done
- Mamba (state-space model, true SSM)   — this test
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from archscope import probes, sae, neurons
from archscope.backends import Backend


MAMBA_NAME = "state-spaces/mamba-130m-hf"


def tokenize(tokenizer, texts: list[str]) -> dict:
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


def main():
    t_start = time.time()
    print(f"[setup] Loading {MAMBA_NAME}…")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MAMBA_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MAMBA_NAME, dtype=torch.float32)
    model.eval()
    print(f"[setup] Loaded. n_layer={model.config.num_hidden_layers}, hidden_size={model.config.hidden_size}")

    # ----- Test 1: Backend extract -----
    print("\n[test 1] MambaBackend.extract — pull residual stream")
    backend = Backend.for_model(model, hint="mamba")
    inputs = tokenize(tokenizer, ["The cat sat on the mat.", "Solve for x: 2x equals 8."])
    records = backend.extract(inputs, layers=["layer_5.residual"])
    rec = records[0]
    print(f"  layer: {rec.layer_name}, shape={tuple(rec.activations.shape)}, meta={rec.meta}")
    assert rec.activations.dim() == 3
    print("  ✓ PASS")

    # ----- Test 2: Probe (math vs non-math) -----
    print("\n[test 2] Probe fit — math vs non-math via layer 12")
    math_texts = [
        "Solve for x: 2x + 3 = 11.", "Compute the derivative of x cubed.",
        "The integral from 0 to 1 of x dx.", "Triangle angles 30 60 90.",
        "The eigenvalue of matrix M.", "By chain rule d/dx sin(x^2).",
        "Find roots of x^2 minus 5x plus 6.", "The Taylor series of e^x.",
    ]
    nonmath_texts = [
        "The cat sat on the mat.", "Music has the power to move.",
        "Mountains stretch to horizon.", "She whispered her secret.",
        "Birds sing at dawn.", "The chef prepared dinner.",
        "Children laughed in the park.", "Rain pattered the window.",
    ]
    inputs_m = tokenize(tokenizer, math_texts)
    inputs_n = tokenize(tokenizer, nonmath_texts)
    pf = probes.fit_probe(
        model, inputs_m, inputs_n,
        layer_name="layer_12.residual", backend_hint="mamba",
    )
    print(f"  train_auroc: {pf.metrics['train_auroc']:.3f}, val_auroc: {pf.metrics['val_auroc']:.3f}")
    assert pf.metrics["train_auroc"] > 0.6
    print("  ✓ PASS")

    # ----- Test 3: Dense SAE -----
    print("\n[test 3] Dense SAE on Mamba residual stream")
    diverse_texts = math_texts + nonmath_texts + [
        "Prove the sum of two evens is even.", "Cauchy-Schwarz inequality states.",
        "Let f be continuous on [0,1].", "Define limit as n to infinity.",
        "By induction on natural numbers.", "The dot product of u and v.",
        "A group is abelian if commutative.", "Fundamental theorem of calculus.",
    ]
    inputs_div = tokenize(tokenizer, diverse_texts)
    rec_div = backend.extract(inputs_div, layers=["layer_12.residual"])[0]
    acts = rec_div.activations.reshape(-1, model.config.hidden_size).detach()
    print(f"  training on {len(acts)} vectors (d={model.config.hidden_size})")
    cfg = sae.SAEConfig(input_dim=model.config.hidden_size, n_features=512, sae_type="dense",
                         sparsity=1e-4, learning_rate=3e-3)
    trained = sae.fit_sae(acts, cfg, epochs=100)
    m = trained.last_metrics
    print(f"  recon: {m['recon']:.4f}, l1: {m['l1']:.4f}, n_active: {m['n_active']:.3f}")
    assert torch.isfinite(torch.tensor(m["recon"]))
    print("  ✓ PASS")

    # ----- Test 4: Rank-1 SAE (designed for recurrent state — Mamba IS recurrent) -----
    print("\n[test 4] Rank-1 SAE on Mamba (recurrent SSM — should suit WriteSAE)")
    cfg2 = sae.SAEConfig(input_dim=model.config.hidden_size, n_features=512, sae_type="rank1",
                          sparsity=1e-4, learning_rate=3e-3)
    trained2 = sae.fit_sae(acts, cfg2, epochs=100)
    m2 = trained2.last_metrics
    print(f"  recon: {m2['recon']:.4f}, l1: {m2['l1']:.4f}, n_active: {m2['n_active']:.3f}")
    print(f"  → comparison: dense={m['recon']:.4f}, rank-1={m2['recon']:.4f} "
          f"({'rank-1 BETTER' if m2['recon'] < m['recon'] else 'dense BETTER'})")
    assert torch.isfinite(torch.tensor(m2["recon"]))
    print("  ✓ PASS")

    # ----- Test 5: Neuron discovery -----
    print("\n[test 5] Neuron discovery on Mamba")
    edit = neurons.find_neurons(
        model, inputs_harmful=inputs_m, inputs_benign=inputs_n,
        config=neurons.NeuronEditConfig(top_frac=0.001),
        backend_hint="mamba",
    )
    total = sum(len(idx) for idx in edit.layer_to_indices.values())
    print(f"  n_layers: {len(edit.layer_to_indices)}, total neurons: {total}")
    assert total > 0
    print("  ✓ PASS")

    # ----- Test 6: Apply neuron edit hook -----
    print("\n[test 6] Neuron edit hook propagates on Mamba blocks")
    test_input = tokenize(tokenizer, ["Compute 2 plus 2 equals"])
    with torch.no_grad():
        base = model(**test_input).logits[:, -1, :].clone()
    edit.multiplier = 0.0
    with edit.apply_hook(model):
        with torch.no_grad():
            modified = model(**test_input).logits[:, -1, :].clone()
    diff = (base - modified).abs().mean().item()
    print(f"  mean |Δ logits|: {diff:.6f}")
    assert diff > 1e-6
    print("  ✓ PASS")

    print(f"\n{'='*60}\nMAMBA INTEGRATION: 6/6 PASS in {time.time()-t_start:.1f}s\n{'='*60}")
    print(f"\nNow we have THREE architectures working:")
    print(f"  • Pythia-160m   (standard transformer, 12 layers, d=768)")
    print(f"  • kazdov-α      (hybrid MoBE-BCN+MHA, 12 layers, d=512)")
    print(f"  • Mamba-130m    (true SSM,           24 layers, d=768)")


if __name__ == "__main__":
    main()
