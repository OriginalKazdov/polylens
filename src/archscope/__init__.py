"""archscope — cross-architecture mechanistic interpretability workbench.

Core methods (architecture-agnostic):
- probes:    linear/MLP probes on hidden states (Drop the Act-style)
- sae:       Dense + Rank-1 sparse autoencoders (WriteSAE-style)
- neurons:   contrastive neuron modulation
- attribute: activation patching + DIM decomposition
- circuits:  induction head, copy, attention-concentration detectors
- lens:      logit lens + tuned lens (Belrose et al 2023)
- diff:      base vs fine-tuned model comparison

Experiment infrastructure:
- backends:  unified extraction API across architectures
- transfer:  cross-arch probe transfer via paired-activation alignment
- bench:     InterpProfile standardized benchmark
- loader:    one-call HuggingFace model + tokenizer + backend loader

Backends: ``transformer``, ``mamba`` (incl. ssm_state), ``kazdov``, ``recurrent``.

Quick start::

    import archscope as ai
    model, tok, backend = ai.load_model("EleutherAI/pythia-160m", arch="transformer")
    result = ai.lens.logit_lens(model, tok, "The capital of France is", target_token=" Paris")
    print(result.to_markdown())
"""

__version__ = "0.2.4"

from . import probes, sae, neurons, attribute, backends, circuits, transfer, bench, lens, diff
from .loader import load_model, make_tokenize_fn

# Kazdov backend registers itself on import — optional, only if kazdov repo present
try:
    from . import kazdov_backend  # noqa: F401
except ImportError:
    pass

__all__ = [
    "probes", "sae", "neurons", "attribute", "backends",
    "circuits", "transfer", "bench", "lens", "diff",
    "load_model", "make_tokenize_fn",
    "__version__",
]
