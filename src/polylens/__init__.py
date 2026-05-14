"""polylens: unified mech interp toolkit across small + RNN + transformer.

Four core methods unified under a single API:
- probes: linear/MLP probes over hidden states (Drop the Act inspired)
- sae: sparse autoencoders for residual + recurrent state (WriteSAE)
- neurons: targeted neuron modulation via contrastive search (Nous Research)
- attribute: activation patching + DIM decomposition (Multi-Agent Sycophancy)

Each method exposes the same architecture-agnostic API:
- .extract(model, inputs) -> hidden states / activations
- .fit(activations, labels) -> learned tool
- .apply(model, inputs) -> modified outputs / scores / explanations

Designed for cross-architecture comparison: transformer, Mamba/SSM, custom RNN.
"""

__version__ = "0.2.0"

from . import probes, sae, neurons, attribute, backends, circuits, transfer, bench, lens, diff

# Kazdov backend registers itself on import — optional, only if kazdov repo present
try:
    from . import kazdov_backend  # noqa: F401
except ImportError:
    pass

__all__ = [
    "probes", "sae", "neurons", "attribute", "backends",
    "circuits", "transfer", "bench", "lens", "__version__",
]
