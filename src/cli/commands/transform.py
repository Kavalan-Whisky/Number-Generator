"""
CLI transform subcommands for sequence transformation.
"""

import json
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _read_data(input_file: Optional[str], data_str: Optional[str]) -> List[float]:
    if input_file:
        try:
            with open(input_file, "r") as f:
                content = f.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [float(x) for x in data]
            except json.JSONDecodeError:
                pass
            return [float(x.strip()) for x in content.replace("\n", ",").split(",") if x.strip()]
        except FileNotFoundError:
            console.print(f"[red]File not found: {input_file}[/red]")
            sys.exit(1)
    elif data_str:
        return [float(x.strip()) for x in data_str.split(",") if x.strip()]
    else:
        console.print("[red]Must provide --input or --data[/red]")
        sys.exit(1)


@click.group()
def transform():
    """Transform and manipulate number sequences."""
    pass


@transform.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
@click.option("--operation", "-o", default="normalize",
              type=click.Choice([
                  "normalize", "standardize", "sort", "reverse", "unique",
                  "running_sum", "running_product", "running_max", "running_min",
                  "differences", "delta_encode", "delta_decode",
                  "moving_average", "ema"
              ]))
@click.option("--window", type=int, default=3, help="Window size for moving averages")
@click.option("--order", type=int, default=1, help="Order for differences")
@click.option("--alpha", type=float, default=0.3, help="Alpha for EMA")
@click.option("--format", "fmt", default="csv",
              type=click.Choice(["table", "json", "csv", "plain"]))
def apply(input_file: Optional[str], data: Optional[str], operation: str,
          window: int, order: int, alpha: float, fmt: str):
    """Apply a transformation to a number sequence."""
    from src.core.transformers.sequence_transformer import (
        normalize, standardize, sort_sequence, reverse, unique,
        running_sum, running_product, running_max, running_min,
        differences, delta_encode, delta_decode,
        moving_average, exponential_moving_average
    )

    numbers = _read_data(input_file, data)

    ops = {
        "normalize": lambda d: normalize(d),
        "standardize": lambda d: standardize(d),
        "sort": lambda d: sort_sequence(d),
        "reverse": lambda d: reverse(d),
        "unique": lambda d: unique(d),
        "running_sum": running_sum,
        "running_product": running_product,
        "running_max": running_max,
        "running_min": running_min,
        "differences": lambda d: differences(d, order),
        "delta_encode": delta_encode,
        "delta_decode": delta_decode,
        "moving_average": lambda d: moving_average(d, window),
        "ema": lambda d: exponential_moving_average(d, alpha),
    }

    try:
        result = ops[operation](numbers)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps([round(x, 6) if isinstance(x, float) else x for x in result]))
    elif fmt == "csv":
        click.echo(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in result))
    elif fmt == "plain":
        for x in result:
            click.echo(f"{x:.4f}" if isinstance(x, float) else str(x))
    else:
        table = Table(title=f"Transform: {operation}", show_header=True)
        table.add_column("Index", style="dim", justify="right")
        table.add_column("Original", justify="right")
        table.add_column("Transformed", style="green", justify="right")

        for i, (orig, trans) in enumerate(zip(numbers[:len(result)], result)):
            table.add_row(
                str(i + 1),
                f"{orig:.4f}" if isinstance(orig, float) else str(orig),
                f"{trans:.4f}" if isinstance(trans, float) else str(trans)
            )
        console.print(table)


@transform.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
@click.option("--format", "fmt", default="binary",
              type=click.Choice(["binary", "octal", "hex", "roman", "scientific", "words", "morse"]))
def format_numbers(input_file: Optional[str], data: Optional[str], fmt: str):
    """Format numbers in various representations."""
    from src.core.transformers.number_formatter import NumberFormatter

    numbers = _read_data(input_file, data)

    try:
        formatted = NumberFormatter.format_sequence(numbers, fmt)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    table = Table(title=f"Numbers as {fmt}", show_header=True)
    table.add_column("Original", justify="right")
    table.add_column(fmt.title(), style="cyan")
    for orig, f in zip(numbers, formatted):
        table.add_row(str(orig), str(f))
    console.print(table)


@transform.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
@click.option("--encoding", default="base64",
              type=click.Choice(["base64", "morse", "rle"]))
@click.option("--decode", is_flag=True, help="Decode instead of encode")
def encode(input_file: Optional[str], data: Optional[str], encoding: str, decode: bool):
    """Encode number sequences in various formats."""
    from src.core.transformers.encoder import Encoder

    if decode:
        # Read string input
        if input_file:
            with open(input_file, "r") as f:
                raw = f.read().strip()
        else:
            raw = data or ""
        if encoding == "base64":
            result = Encoder.base64_decode(raw)
            click.echo(json.dumps(result))
        elif encoding == "morse":
            result = Encoder.morse_decode(raw)
            click.echo(json.dumps(result))
        else:
            console.print("[yellow]Decode not supported for RLE via CLI[/yellow]")
    else:
        numbers = _read_data(input_file, data)
        int_numbers = [int(x) for x in numbers]
        if encoding == "base64":
            result = Encoder.base64_encode(int_numbers)
            click.echo(result)
        elif encoding == "morse":
            result = Encoder.morse_encode(int_numbers)
            click.echo(result)
        elif encoding == "rle":
            result = Encoder.rle_encode(int_numbers)
            click.echo(json.dumps(result))
