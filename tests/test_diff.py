"""Test Model Diff — find what changes between base and fine-tuned variants.

Approach:
1. Identity sanity check: diff(model, model) → drift ≈ 0 everywhere
2. Perturbation: deep-copy model, perturb a specific layer's weights → drift
   concentrated at that layer.
"""
from __future__ import annotations
import sys
import time
import copy
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from archscope import diff


def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[1] Identity check: diff(model, model) → near-zero drift")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", dtype=torch.float32)
    base.eval()

    cal = [
        "The cat sat on the mat.",
        "Solve for x: 2x = 10.",
        "Music makes me happy.",
        "Compute derivative of x^2.",
        "Mountains stretch far.",
        "Find the integral.",
        "Children laughed loudly.",
        "Prove the theorem.",
    ]
    d_identity = diff.compare(base, base, tok, cal, backend_hint="transformer",
                                run_circuits=False)
    max_drift = max(l.relative_drift for l in d_identity.layer_drift)
    max_l2 = max(l.mean_l2_delta for l in d_identity.layer_drift)
    print(f"  max relative_drift = {max_drift:.2e}, max ||Δa|| = {max_l2:.2e}")
    assert max_drift < 1e-5, f"Identity diff should be ~0, got {max_drift}"
    print("  ✓ PASS — identity diff produces near-zero drift")

    print("\n[2] Perturbation check: shift layer-5 MLP weights → drift concentrates there")
    perturbed = copy.deepcopy(base)
    perturbed.eval()
    # Inject noise into layer 5's MLP weights
    target_layer = perturbed.gpt_neox.layers[5].mlp.dense_h_to_4h
    with torch.no_grad():
        noise = 0.05 * torch.randn_like(target_layer.weight)
        target_layer.weight.add_(noise)

    d_perturbed = diff.compare(base, perturbed, tok, cal, backend_hint="transformer",
                                 run_circuits=False)
    print(f"  Layer-by-layer relative drift:")
    for ld in d_perturbed.layer_drift:
        print(f"    L{ld.layer:>2}: {ld.relative_drift:.4f}  (cos sim {ld.cosine_similarity:.4f})")

    # Drift should be near-zero before layer 5 and substantial at/after layer 5
    pre_drift = sum(l.relative_drift for l in d_perturbed.layer_drift if l.layer < 5) / 5
    post_drift = sum(l.relative_drift for l in d_perturbed.layer_drift if l.layer >= 5) / max(1, len(d_perturbed.layer_drift) - 5)
    print(f"  → pre-layer-5 avg drift: {pre_drift:.4f}")
    print(f"  → post-layer-5 avg drift: {post_drift:.4f}")
    assert pre_drift < 1e-4, f"Pre-perturbation layers should be ~0 drift, got {pre_drift}"
    assert post_drift > pre_drift * 10, f"Post-layer drift should be much higher than pre"
    print("  ✓ PASS — drift correctly concentrates at/after perturbation site")

    # Show top shifted neurons at the perturbed layer
    perturbed_layer_drift = next((l for l in d_perturbed.layer_drift if l.layer == 5), None)
    if perturbed_layer_drift:
        top = perturbed_layer_drift.top_shifted_neurons[:5]
        print(f"  Top-5 shifted neurons at layer 5: {top}")

    # Print markdown report sample
    print(f"\n--- Markdown sample (first 15 lines) ---")
    md = d_perturbed.to_markdown()
    for line in md.split("\n")[:15]:
        print(line)

    print(f"\n{'='*64}\nMODEL DIFF: 2/2 PASS in {time.time()-t_start:.1f}s\n{'='*64}")


if __name__ == "__main__":
    main()
