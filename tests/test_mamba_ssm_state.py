"""Test: SSM state extraction from Mamba — the technical differentiator.

Verifies:
- New `.ssm_state` layer suffix works
- Shape is (batch, intermediate_size, ssm_state_size) per layer
- Different texts produce different SSM states (state actually encodes content)
- SAE on SSM state trains successfully (unique to this library)
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, "/Users/kazdov/code/OriginalKazdov/archscope/src")

from archscope import sae
from archscope.backends import Backend


def tokenize(tok, texts):
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[setup] Loading mamba-130m-hf…")
    tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float32)
    model.eval()
    backend = Backend.for_model(model, hint="mamba")

    # ----- Test 1: layer_names includes both .residual AND .ssm_state -----
    print("\n[test 1] Backend.layer_names includes .ssm_state entries")
    names = backend.layer_names()
    residual_names = [n for n in names if ".residual" in n]
    ssm_names = [n for n in names if ".ssm_state" in n]
    print(f"  residual layers: {len(residual_names)}, ssm_state layers: {len(ssm_names)}")
    assert len(residual_names) == 24 and len(ssm_names) == 24
    print("  ✓ PASS")

    # ----- Test 2: SSM state extraction returns correct shape -----
    print("\n[test 2] Extract layer_5.ssm_state — shape (B, intermediate_size, ssm_state_size)")
    inputs = tokenize(tok, ["The cat sat on the mat.", "Solve x squared equals 16."])
    records = backend.extract(inputs, layers=["layer_5.ssm_state"])
    assert len(records) == 1
    rec = records[0]
    print(f"  shape: {tuple(rec.activations.shape)}")
    print(f"  meta: {rec.meta}")
    B, D_inner, D_state = rec.activations.shape
    assert B == 2 and D_inner == 1536 and D_state == 16
    print(f"  ✓ PASS — total SSM dims per example: {D_inner * D_state} (= {D_inner} × {D_state})")

    # ----- Test 3: Different inputs produce different SSM states -----
    print("\n[test 3] Different texts produce DIFFERENT ssm_states")
    rec_math = backend.extract(tokenize(tok, ["Solve x squared equals 16."]), layers=["layer_5.ssm_state"])[0]
    rec_nature = backend.extract(tokenize(tok, ["Birds fly south in winter."]), layers=["layer_5.ssm_state"])[0]
    state_diff = (rec_math.activations - rec_nature.activations).abs().mean().item()
    print(f"  |state_math - state_nature|.mean() = {state_diff:.4f}")
    assert state_diff > 0.001, "States are identical — extraction broken"
    print("  ✓ PASS — SSM state encodes content")

    # ----- Test 4: Train SAE on SSM state (UNIQUE to our library) -----
    print("\n[test 4] Train SAE on SSM state (flattened: intermediate × state = features)")
    diverse_texts = [
        "The cat sat on the mat softly.", "Solve x + 5 = 12 here.",
        "Mountains stretch to horizon line.", "Integrate x squared dx.",
        "Music inspires the lonely heart.", "Find the derivative now.",
        "Children laughed in the park.", "Triangle has angles 60 60 60.",
        "Rain fell softly all afternoon.", "Eigenvalue is two times one.",
        "Coffee aroma filled the kitchen.", "Prove the theorem carefully.",
    ]
    inputs_div = tokenize(tok, diverse_texts)
    rec = backend.extract(inputs_div, layers=["layer_5.ssm_state"])[0]
    # Flatten (B, d_inner, d_state) → (B, d_inner * d_state) as feature vectors
    acts = rec.activations.reshape(rec.activations.shape[0], -1).detach()
    print(f"  training on {len(acts)} ssm_state vectors (d={acts.shape[-1]})")
    cfg = sae.SAEConfig(input_dim=acts.shape[-1], n_features=1024, sae_type="dense",
                         sparsity=1e-4, learning_rate=3e-3)
    trained = sae.fit_sae(acts, cfg, epochs=80)
    m = trained.last_metrics
    print(f"  recon: {m['recon']:.4f}, l1: {m['l1']:.6f}, n_active: {m['n_active']:.3f}")
    assert torch.isfinite(torch.tensor(m["recon"]))
    print("  ✓ PASS — SAE trains on SSM state successfully")

    # ----- Test 5: Mixed extraction — request both .residual and .ssm_state -----
    print("\n[test 5] Mixed: request residual + ssm_state for same layer simultaneously")
    records = backend.extract(inputs, layers=["layer_8.residual", "layer_8.ssm_state"])
    assert len(records) == 2
    res = next(r for r in records if r.meta["kind"] == "residual")
    ssm = next(r for r in records if r.meta["kind"] == "ssm_state")
    print(f"  residual:  {tuple(res.activations.shape)}")
    print(f"  ssm_state: {tuple(ssm.activations.shape)}")
    assert res.activations.shape[0] == ssm.activations.shape[0]
    print("  ✓ PASS — mixed extraction works")

    print(f"\n{'='*64}\nSSM STATE EXTRACTION: 5/5 PASS in {time.time()-t_start:.1f}s\n{'='*64}")
    print("\nWe now expose Mamba's SSM hidden state (h_t) for mech interp.")
    print("This is unique among open-source mech interp libraries.")


if __name__ == "__main__":
    main()
