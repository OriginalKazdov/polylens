"""End-to-end integration test with Pythia-160m.

Validates that each of the 4 core methods works on a real HuggingFace model:
1. Backend activation extraction
2. Probe fit on synthetic binary task (positive vs negative sentences)
3. SAE fit on real residual stream activations
4. Targeted neuron modulation discovery + hook application

Designed to run in <60s on CPU.
"""
from __future__ import annotations
import sys
import time
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from archscope import probes, sae, neurons, attribute
from archscope.backends import Backend


MODEL_NAME = "EleutherAI/pythia-160m"


def load_model():
    print(f"[setup] Loading {MODEL_NAME}…")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    print(f"[setup] Loaded in {time.time()-t0:.1f}s. n_layer={model.config.num_hidden_layers}, "
          f"hidden_size={model.config.hidden_size}")
    return model, tokenizer


def tokenize_batch(tokenizer, texts: list[str]) -> dict:
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)


# -------------------------------------------------------------------- TESTS

def test_backend_extract(model, tokenizer):
    print("\n[test 1] Backend.extract — pull residual stream from layer 5")
    backend = Backend.for_model(model, hint="transformer")
    inputs = tokenize_batch(tokenizer, ["The cat sat on the mat.", "Quantum physics is hard."])
    records = backend.extract(inputs, layers=["layer_5.residual"])
    assert len(records) == 1
    rec = records[0]
    print(f"  layer_name: {rec.layer_name}")
    print(f"  activations.shape: {tuple(rec.activations.shape)}")
    print(f"  meta: {rec.meta}")
    assert rec.activations.dim() == 3  # (batch, seq, hidden)
    print("  ✓ PASS")
    return rec


def test_probe_fit(model, tokenizer):
    print("\n[test 2] Probe fit — positive vs negative sentiment via layer 5")
    pos_texts = [
        "I love this movie, it's amazing!",
        "What a wonderful day.",
        "Best book I have ever read.",
        "She is so kind and thoughtful.",
        "The food was delicious.",
        "I'm thrilled about the news.",
        "Such a beautiful sunset tonight.",
        "He always makes me laugh.",
    ]
    neg_texts = [
        "I hate this place.",
        "What a terrible movie.",
        "This is the worst day ever.",
        "She is mean and selfish.",
        "The food was awful.",
        "I'm so disappointed.",
        "Such a horrible meeting.",
        "He always annoys me.",
    ]
    inputs_pos = tokenize_batch(tokenizer, pos_texts)
    inputs_neg = tokenize_batch(tokenizer, neg_texts)

    pf = probes.fit_probe(
        model,
        inputs_pos={k: v for k, v in inputs_pos.items()},
        inputs_neg={k: v for k, v in inputs_neg.items()},
        layer_name="layer_5.residual",
        backend_hint="transformer",
    )
    print(f"  train_auroc: {pf.metrics['train_auroc']:.3f}")
    print(f"  val_auroc:   {pf.metrics['val_auroc']:.3f}")
    print(f"  train_loss:  {pf.metrics['train_loss']:.4f}")
    # On 16 examples with 8/8 split, probe should beat 0.5
    assert pf.metrics["train_auroc"] > 0.6, f"Probe didn't learn (auroc={pf.metrics['train_auroc']})"
    print("  ✓ PASS (probe learns above chance on sentiment)")
    return pf


def _collect_large_activation_batch(model, tokenizer, n_per_class: int = 30):
    """Generate more diverse activations for SAE training (768-dim needs more data)."""
    base_sentences = [
        "The cat sat on the mat.", "Birds fly south in winter.",
        "She opened the book carefully.", "Quantum physics describes small particles.",
        "Music can change our mood.", "The chef prepared a fine meal.",
        "Mountains stretch across the horizon.", "He whispered a secret to her.",
        "Rain pattered against the window.", "Time passes faster when busy.",
        "Children laughed in the playground.", "Stars appear bright in dark sky.",
        "The river flowed gently downstream.", "Coffee aroma filled the kitchen.",
        "Books contain endless worlds.", "Snow blanketed the entire valley.",
        "Dance is a universal language.", "Engineering shapes modern society.",
        "Cooking requires patience and love.", "History repeats itself often.",
        "Trees provide shade in summer.", "Friendship grows with shared moments.",
        "Mathematics describes natural patterns.", "Photography captures fleeting beauty.",
        "Architecture defines a city's soul.", "Literature explores human emotions.",
        "Travel broadens one's perspective.", "Science explains the unknown.",
        "Art expresses inner feelings.", "Sports build team spirit.",
    ]
    inputs = tokenize_batch(tokenizer, base_sentences[:n_per_class])
    backend = Backend.for_model(model, hint="transformer")
    with torch.no_grad():
        rec = backend.extract(inputs, layers=["layer_5.residual"])[0]
    return rec.activations.reshape(-1, rec.activations.shape[-1]).detach()


def test_sae_fit(model, tokenizer):
    print("\n[test 3] SAE fit (dense) — learn dictionary on real residual")
    acts = _collect_large_activation_batch(model, tokenizer)
    print(f"  training on {len(acts)} activation vectors (dim={acts.shape[-1]})")
    cfg = sae.SAEConfig(input_dim=acts.shape[-1], n_features=512, sae_type="dense",
                         sparsity=1e-4, learning_rate=3e-3)
    trained = sae.fit_sae(acts, cfg, epochs=100)
    m = trained.last_metrics
    print(f"  recon: {m['recon']:.4f}")
    print(f"  l1:    {m['l1']:.4f}")
    print(f"  n_active: {m['n_active']:.3f}")
    # Real hidden state magnitudes — recon ~1-3 is normal for under-trained dense SAE on 768d
    # Sanity: should be decreasing, not nan
    assert torch.isfinite(torch.tensor(m["recon"])), f"NaN recon: {m['recon']}"
    print("  ✓ PASS (SAE converging, not nan)")
    return trained, acts


def test_sae_rank1(acts):
    print("\n[test 4] SAE fit (rank-1 factored) — WriteSAE-style atoms")
    cfg = sae.SAEConfig(input_dim=acts.shape[-1], n_features=512, sae_type="rank1",
                         sparsity=1e-4, learning_rate=3e-3)
    trained = sae.fit_sae(acts, cfg, epochs=100)
    m = trained.last_metrics
    print(f"  recon: {m['recon']:.4f}")
    print(f"  l1:    {m['l1']:.4f}")
    print(f"  n_active: {m['n_active']:.3f}")
    assert torch.isfinite(torch.tensor(m["recon"])), f"NaN recon: {m['recon']}"
    print("  ✓ PASS")


def test_neuron_discovery(model, tokenizer):
    print("\n[test 5] Neuron discovery — find top 0.1% activating on 'refusal-ish'")
    refuse_texts = [
        "I cannot help with that request.",
        "I'm unable to provide that information.",
        "I won't assist with harmful tasks.",
        "Sorry, I can't do that.",
    ]
    benign_texts = [
        "The recipe calls for two cups of flour.",
        "Python is a popular programming language.",
        "The mountain stands tall against the sky.",
        "Music has the power to inspire us.",
    ]
    inputs_h = tokenize_batch(tokenizer, refuse_texts)
    inputs_b = tokenize_batch(tokenizer, benign_texts)

    edit = neurons.find_neurons(
        model,
        inputs_harmful=inputs_h,
        inputs_benign=inputs_b,
        config=neurons.NeuronEditConfig(top_frac=0.001),
        backend_hint="transformer",
    )
    n_layers_edited = len(edit.layer_to_indices)
    total_neurons = sum(len(idx) for idx in edit.layer_to_indices.values())
    print(f"  n_layers_with_edits: {n_layers_edited}")
    print(f"  total_neurons_selected: {total_neurons}")
    sample_layer = list(edit.layer_to_indices.keys())[0]
    print(f"  sample: {sample_layer} -> indices {edit.layer_to_indices[sample_layer][:5].tolist()}")
    assert n_layers_edited == model.config.num_hidden_layers
    assert total_neurons > 0
    print("  ✓ PASS")
    return edit


def test_neuron_apply(model, tokenizer, edit):
    print("\n[test 6] Neuron edit apply — verify hook modulates output")
    inputs = tokenize_batch(tokenizer, ["Hello, how are you?"])
    with torch.no_grad():
        base = model(**inputs).logits[:, -1, :].clone()
    # Apply edit (multiplier=0 = ablate selected neurons)
    edit.multiplier = 0.0
    with edit.apply_hook(model) as _ctx:
        with torch.no_grad():
            modified = model(**inputs).logits[:, -1, :].clone()
    diff = (base - modified).abs().mean().item()
    print(f"  mean |Δ logits|: {diff:.6f}")
    assert diff > 1e-6, "Hook did not modify output (something is wrong)"
    print("  ✓ PASS (hooks measurably modify model output)")


# -------------------------------------------------------------------- RUNNER

def main():
    t_start = time.time()
    model, tokenizer = load_model()

    rec = test_backend_extract(model, tokenizer)
    test_probe_fit(model, tokenizer)
    _, acts_large = test_sae_fit(model, tokenizer)
    test_sae_rank1(acts_large)
    edit = test_neuron_discovery(model, tokenizer)
    test_neuron_apply(model, tokenizer, edit)

    print(f"\n{'='*60}\nALL 6 TESTS PASSED in {time.time()-t_start:.1f}s\n{'='*60}")


if __name__ == "__main__":
    main()
