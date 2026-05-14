"""Command-line interface for archscope."""
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
    """archscope — cross-architecture mechanistic interpretability toolkit."""


@cli.command()
def info() -> None:
    """Show available methods + supported backends."""
    methods = Table(title=f"archscope v{__version__}")
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
@click.option("--out", default=None,
              help="Output file. Format inferred from extension: .json or .md. "
                   "Without --out, prints markdown to stdout.")
def bench(model_name: str, arch: str, out: str | None) -> None:
    """Run archscope InterpBench on a HuggingFace model.

    Examples:
      archscope bench EleutherAI/pythia-160m --arch transformer
      archscope bench EleutherAI/pythia-160m --arch transformer --out pythia.md
      archscope bench state-spaces/mamba-130m-hf --arch mamba --out mamba.json
    """
    # Lazy imports keep `archscope info` fast (no torch/transformers).
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

    markdown = bench_mod.profile_to_markdown(profile)
    console.print()
    console.print(markdown)

    if out:
        from dataclasses import asdict
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ext = os.path.splitext(out)[1].lower()
        if ext == ".md":
            with open(out, "w") as f:
                f.write(markdown + "\n")
            console.print(f"\n[green]Saved markdown report to {out}[/green]")
        elif ext == ".json" or ext == "":
            with open(out, "w") as f:
                json.dump(asdict(profile), f, indent=2, default=str)
            console.print(f"\n[green]Saved JSON profile to {out}[/green]")
        else:
            raise click.UsageError(
                f"Unsupported --out extension '{ext}'. Use .md or .json."
            )


if __name__ == "__main__":
    cli()
