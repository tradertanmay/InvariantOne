"""
InvariantOne Command-Line Interface.

Supports human-readable and JSON formatted decision evaluations, model verification,
and optional local HTTP serving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invariantone",
        description="InvariantOne: High-Assurance Permutation-Equivariant Natural-Language Decision Model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # decide subcommand
    decide_parser = subparsers.add_parser("decide", help="Evaluate runtime options for a given state and question")
    decide_parser.add_argument("--state", type=str, required=True, help="State description")
    decide_parser.add_argument("--question", type=str, required=True, help="Decision question")
    decide_parser.add_argument(
        "--option",
        type=str,
        action="append",
        dest="options",
        required=True,
        help="Candidate option (specify multiple times for K options)",
    )
    decide_parser.add_argument("--model-path", type=str, default="invariantone-v1", help="Path to checkpoint directory")
    decide_parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda, mps, cpu)")
    decide_parser.add_argument("--json", action="store_true", dest="json_output", help="Output in machine-readable JSON")
    decide_parser.add_argument("--strict", action="store_true", default=True, help="Enforce strict hash checks")

    # info subcommand
    info_parser = subparsers.add_parser("info", help="Display frozen model metadata and hashes")
    info_parser.add_argument("--json", action="store_true", dest="json_output", help="Output in JSON")

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify cryptographic checkpoint integrity")
    verify_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to checkpoint directory")

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Run local HTTP inference service")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")
    serve_parser.add_argument("--model-path", type=str, default="invariantone-v1", help="Path to checkpoint directory")
    serve_parser.add_argument("--device", type=str, default="auto", help="Execution device")

    return parser


def handle_decide(args: argparse.Namespace) -> int:
    from invariantone.model import InvariantOne

    if not args.options or len(args.options) < 2:
        if args.json_output:
            print(json.dumps({"error": "At least 2 options are required"}, indent=2))
        else:
            console.print("[bold red]Error:[/bold red] At least 2 options are required.")
        return 1

    try:
        model = InvariantOne.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            device=args.device,
            strict=args.strict,
        )
        result = model.decide(
            state=args.state,
            question=args.question,
            options=args.options,
        )
    except Exception as e:
        if args.json_output:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(f"[bold red]Inference Error:[/bold red] {e}")
        return 1

    if args.json_output:
        out = {
            "choice_index": result.choice_index,
            "choice": result.choice,
            "probabilities": [round(p, 4) for p in result.probabilities],
            "scores": [round(s, 4) for s in result.scores],
            "options": result.options,
            "latency_ms": result.latency_ms,
            "model": "InvariantOne-v1",
        }
        print(json.dumps(out, indent=2))
    else:
        console.print("\n[bold cyan]InvariantOne Decision Result[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Idx", justify="center", width=4)
        table.add_column("Option", justify="left")
        table.add_column("Probability", justify="right", width=12)
        table.add_column("Raw Score", justify="right", width=12)
        table.add_column("Selection", justify="center", width=10)

        for i, opt in enumerate(result.options):
            is_chosen = (i == result.choice_index)
            sel_str = "[bold green]★ CHOSEN[/bold green]" if is_chosen else ""
            opt_style = "bold green" if is_chosen else ""
            table.add_row(
                str(i),
                f"[{opt_style}]{opt}[/{opt_style}]" if opt_style else opt,
                f"{result.probabilities[i]:.2%}",
                f"{result.scores[i]:+.4f}",
                sel_str,
            )

        console.print(table)
        console.print(f"[dim]Device: {result.device} | Latency: {result.latency_ms:.1f}ms | Temp: {result.temperature}[/dim]\n")

    return 0


def handle_info(args: argparse.Namespace) -> int:
    from invariantone.config import InvariantOneConfig

    cfg = InvariantOneConfig()
    info_dict = {
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "research_designation": cfg.research_designation,
        "backbone": cfg.backbone_name,
        "backbone_revision": cfg.backbone_revision,
        "architecture": cfg.architecture,
        "lora_rank": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "adapted_layers": cfg.adapted_layers,
        "training_operator_breadth": cfg.training_operator_breadth,
        "calibration_temperature": cfg.calibration_temperature,
        "weights_hashes": {
            "lora_sha256": cfg.lora_sha256,
            "head_sha256": cfg.head_sha256,
        },
    }

    if args.json_output:
        print(json.dumps(info_dict, indent=2))
    else:
        console.print(Panel(
            f"[bold cyan]Model:[/bold cyan] {cfg.model_name} v{cfg.model_version} ({cfg.research_designation})\n"
            f"[bold cyan]Backbone:[/bold cyan] {cfg.backbone_name} (commit {cfg.backbone_revision[:12]}...)\n"
            f"[bold cyan]Architecture:[/bold cyan] {cfg.architecture} (Layers {cfg.adapted_layers}, rank={cfg.lora_r})\n"
            f"[bold cyan]Head Mode:[/bold cyan] {cfg.head_mode} ({cfg.head_param_count:,} params)\n"
            f"[bold cyan]Calibration Temp:[/bold cyan] tau* = {cfg.calibration_temperature}\n"
            f"[bold cyan]LoRA SHA-256:[/bold cyan] {cfg.lora_sha256}\n"
            f"[bold cyan]Head SHA-256:[/bold cyan] {cfg.head_sha256}",
            title="[bold green]InvariantOne v1 Metadata[/bold green]",
        ))
    return 0


def handle_verify(args: argparse.Namespace) -> int:
    from invariantone.loading.checkpoint_loader import resolve_checkpoint_paths
    from invariantone.loading.hash_verification import verify_checkpoint_hash
    from invariantone.config import FROZEN_HEAD_SHA256, FROZEN_LORA_SHA256

    try:
        lora_p, head_p = resolve_checkpoint_paths(pretrained_name_or_path=args.checkpoint_dir)
        console.print(f"Verifying checkpoints in {lora_p.parent}...")

        ok_lora, l_sha = verify_checkpoint_hash(lora_p, FROZEN_LORA_SHA256, "LoRA adapter", strict=True)
        console.print(f"  ✓ LoRA Adapter: {lora_p.name} [{l_sha[:16]}...] [bold green]VERIFIED[/bold green]")

        ok_head, h_sha = verify_checkpoint_hash(head_p, FROZEN_HEAD_SHA256, "Comparative head", strict=True)
        console.print(f"  ✓ Comparative Head: {head_p.name} [{h_sha[:16]}...] [bold green]VERIFIED[/bold green]")

        console.print("\n[bold green]All InvariantOne v1 checkpoint cryptographic signatures verified successfully![/bold green]")
        return 0
    except Exception as e:
        console.print(f"[bold red]Verification Failed:[/bold red] {e}")
        return 1


def handle_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from invariantone.model import InvariantOne
    from invariantone.server.app import create_app

    console.print(f"[bold cyan]Starting InvariantOne service on {args.host}:{args.port}...[/bold cyan]")
    model = InvariantOne.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        device=args.device,
        strict=True,
    )
    app = create_app(model=model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "decide":
        return handle_decide(args)
    elif args.command == "info":
        return handle_info(args)
    elif args.command == "verify":
        return handle_verify(args)
    elif args.command == "serve":
        return handle_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
