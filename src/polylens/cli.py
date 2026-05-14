"""Command-line interface for polylens."""
from __future__ import annotations

import json
import os

import click
from rich.console import Console
from rich.table import Table

from . import __version__

console = Console()


@click.group()
def cli() -> None:
    """polylens — cross-architecture mechanistic interpretability toolkit."""


@cli.command()
def info() -> None:
    """Show available methods + supported backends."""
    methods = Table(title=f"polylens v{__version__}")
    methods.add_column("Method")
    methods.add_column("Module")
    methods.add_column("Source paper")
    for row in (
        ("Probes",              "probes.fit_probe",           "Drop the Act (2605.11467)"),
        ("SAE",                 "sae.fit_sae",                "WriteSAE (2605.12770)"),
        ("Neuron mod",          "neurons.find_neurons",       "Targeted Neuron Mod (2605.12290)"),
        ("Activation patch",    "attribute.activation_patch", "Multi-Agent Sycophancy (2605.12991)"),
        ("Cross-arch transfer", "transfer.evaluate_transfer", "this library"),
        ("Circuit detection",   "circuits.run_all_circuits",  "this library"),
        ("Logit/tuned lens",    "lens.logit_lens",            "Belrose et al 2023"),
        ("Model diff",          "diff.compare",               "this library"),
        ("InterpBench",         "bench.benchmark",            "this library"),
    ):
        methods.add_row(*row)
    console.print(methods)

    backends = Table(title="Backends")
    backends.add_column("Name")
    backends.add_column("Architecture family")
    for row in (
        ("transformer", "HuggingFace decoder LMs (Llama, GPT, Qwen, Pythia, ...)"),
        ("mamba",       "Mamba / Mamba-2 SSM — exposes .ssm_state (recurrent h_t)"),
        ("kazdov",      "Kazdov-α hybrid MoBE-BCN+MHA"),
        ("recurrent",   "Generic RNN (subclass per model)"),
    ):
        backends.add_row(*row)
    console.print(backends)


@cli.command()
@click.argument("model_name")
@click.option("--arch", default="transformer",
              type=click.Choice(["transformer", "mamba", "kazdov"]),
              help="Architecture family.")
@click.option("--out", default=None, help="Output JSON file (default: prints only).")
def bench(model_name: str, arch: str, out: str | None) -> None:
    """Run polylens InterpBench on a HuggingFace model.

    Examples:
      polylens bench EleutherAI/pythia-160m --arch transformer
      polylens bench state-spaces/mamba-130m-hf --arch mamba
    """
    # Lazy imports keep `polylens info` fast (no torch/transformers).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import bench as bench_mod

    console.print(f"[cyan]→ Loading {model_name}…[/cyan]")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    def tokenize_fn(texts):
        return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)

    arch_family = {"transformer": "transformer", "mamba": "ssm", "kazdov": "hybrid"}[arch]
    profile = bench_mod.benchmark(
        model_name=model_name,
        model=model,
        tokenizer=tok,
        backend_hint=arch,
        arch_family=arch_family,
        tokenize_fn=tokenize_fn,
    )
    console.print()
    console.print(bench_mod.profile_to_markdown(profile))

    if out:
        from dataclasses import asdict
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(asdict(profile), f, indent=2, default=str)
        console.print(f"\n[green]Saved profile to {out}[/green]")


if __name__ == "__main__":
    cli()
