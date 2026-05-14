"""CLI for mechinterp-small."""
from __future__ import annotations
import json
import os
import click
from rich.console import Console
from rich.table import Table


console = Console()


@click.group()
def cli():
    """mechinterp-small — unified mech interp toolkit."""
    pass


@cli.command()
def info():
    """Show available methods + supported backends."""
    t = Table(title="mechinterp-small v0.1.0")
    t.add_column("Method"); t.add_column("Module"); t.add_column("Source paper")
    t.add_row("Probes", "probes.fit_probe", "Drop the Act (2605.11467)")
    t.add_row("SAE", "sae.fit_sae", "WriteSAE (2605.12770)")
    t.add_row("Neuron mod", "neurons.find_neurons", "Targeted Neuron Mod (2605.12290)")
    t.add_row("Activation patch", "attribute.activation_patch", "Multi-Agent Sycophancy (2605.12991)")
    t.add_row("Cross-arch transfer", "transfer.evaluate_transfer", "this library")
    t.add_row("Circuit detection", "circuits.run_all_circuits", "this library")
    t.add_row("InterpBench", "bench.benchmark", "this library")
    console.print(t)

    t2 = Table(title="Backends")
    t2.add_column("Name"); t2.add_column("Architecture family")
    t2.add_row("transformer", "HuggingFace decoder LMs (Llama, GPT, Qwen, Pythia)")
    t2.add_row("mamba", "Mamba / Mamba-2 SSM — UNIQUE: exposes .ssm_state")
    t2.add_row("kazdov", "Kazdov-α hybrid MoBE-BCN+MHA")
    t2.add_row("recurrent", "Generic RNN (subclass per model)")
    console.print(t2)


@cli.command()
@click.argument("model_name")
@click.option("--arch", default="transformer", help="Architecture family: transformer|mamba|kazdov")
@click.option("--out", default=None, help="Output JSON file (default: prints only)")
def bench(model_name: str, arch: str, out: str | None):
    """Run InterpBench on a HuggingFace model.

    Example: mechinterp bench EleutherAI/pythia-160m --arch transformer
             mechinterp bench state-spaces/mamba-130m-hf --arch mamba
    """
    from . import bench as bench_mod
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"[cyan]→ Loading {model_name}…[/cyan]")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    def tokenize_fn(texts):
        return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)

    arch_family = {"transformer": "transformer", "mamba": "ssm", "kazdov": "hybrid"}.get(arch, "custom")
    profile = bench_mod.benchmark(
        model_name=model_name,
        model=model, tokenizer=tok,
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
