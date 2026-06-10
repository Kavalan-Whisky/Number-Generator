"""
CLI generate subcommands for number sequence generation.
"""

import json
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
def generate():
    """Generate various types of number sequences."""
    pass


def _display_numbers(numbers: List, title: str = "Generated Numbers", format_: str = "table"):
    """Display numbers in the specified format."""
    if format_ == "json":
        click.echo(json.dumps(numbers))
    elif format_ == "csv":
        click.echo(",".join(str(n) for n in numbers))
    elif format_ == "plain":
        click.echo("\n".join(str(n) for n in numbers))
    else:
        table = Table(title=title, show_header=True)
        table.add_column("Index", style="dim", justify="right")
        table.add_column("Value", style="green", justify="right")
        for i, n in enumerate(numbers):
            table.add_row(str(i + 1), str(n))
        console.print(table)


@generate.command()
@click.option("--algorithm", "-a", default="mersenne",
              type=click.Choice(["lcg", "xorshift32", "xorshift64", "pcg", "lfsr", "bbs", "middle_square", "mersenne"]),
              help="PRNG algorithm to use")
@click.option("--count", "-n", default=10, type=int, help="Number of values to generate")
@click.option("--min", "min_val", default=0, type=int, help="Minimum value (inclusive)")
@click.option("--max", "max_val", default=1000, type=int, help="Maximum value (inclusive)")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "csv", "plain"]),
              help="Output format")
@click.option("--preset", type=str, default=None, help="LCG preset name")
def random(algorithm: str, count: int, min_val: int, max_val: int,
           seed: Optional[int], fmt: str, preset: Optional[str]):
    """Generate random numbers using various PRNG algorithms."""
    from src.core.generators.random_generator import RandomGeneratorFactory

    with console.status(f"Generating {count} random numbers using {algorithm}..."):
        try:
            kwargs = {}
            if preset:
                kwargs["preset"] = preset
            gen = RandomGeneratorFactory.create(algorithm, seed=seed, **kwargs)
            numbers = gen.generate(count, min_val, max_val)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    _display_numbers(numbers, f"Random Numbers ({algorithm})", fmt)
    if fmt == "table":
        console.print(f"\n[dim]Algorithm: {algorithm} | Count: {count} | Range: [{min_val}, {max_val}][/dim]")


@generate.command()
@click.option("--count", "-n", default=20, type=int, help="Number of primes to generate")
@click.option("--start", default=2, type=int, help="Starting value (first prime >= start)")
@click.option("--algorithm", "-a", default="sieve",
              type=click.Choice(["sieve", "sundaram"]),
              help="Sieve algorithm")
@click.option("--test-primality", is_flag=True, help="Show primality test for each number")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "csv", "plain"]))
def prime(count: int, start: int, algorithm: str, test_primality: bool, fmt: str):
    """Generate prime numbers."""
    from src.core.generators.prime_generator import PrimeGeneratorFactory

    with console.status(f"Generating {count} prime numbers..."):
        try:
            primes = PrimeGeneratorFactory.generate_primes(count, start, algorithm)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    _display_numbers(primes, f"Prime Numbers (from {start})", fmt)
    if fmt == "table":
        console.print(f"\n[dim]Count: {count} | Start: {start} | Algorithm: {algorithm}[/dim]")
        if len(primes) >= 2:
            console.print(f"[dim]Range: {primes[0]} - {primes[-1]}[/dim]")


@generate.command()
@click.option("--count", "-n", default=20, type=int, help="Number of Fibonacci values")
@click.option("--variant", default="iterative",
              type=click.Choice(["iterative", "recursive", "matrix", "closed_form"]),
              help="Fibonacci computation method")
@click.option("--start-index", default=0, type=int, help="Starting index")
@click.option("--type", "fib_type", default="fibonacci",
              type=click.Choice(["fibonacci", "lucas", "tribonacci", "tetranacci"]),
              help="Fibonacci variant")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "csv", "plain"]))
def fibonacci(count: int, variant: str, start_index: int, fib_type: str, fmt: str):
    """Generate Fibonacci and related sequences."""
    from src.core.generators.fibonacci_generator import (
        FibonacciGenerator, LucasSequence, GeneralizedFibonacci
    )

    with console.status(f"Generating {count} {fib_type} numbers..."):
        try:
            if fib_type == "fibonacci":
                gen = FibonacciGenerator()
                numbers = gen.generate(count, variant, start_index)
            elif fib_type == "lucas":
                numbers = LucasSequence.generate(count, start_index)
            elif fib_type == "tribonacci":
                gen = GeneralizedFibonacci.tribonacci()
                numbers = gen.generate(count, start_index)
            elif fib_type == "tetranacci":
                gen = GeneralizedFibonacci.tetranacci()
                numbers = gen.generate(count, start_index)
            else:
                raise ValueError(f"Unknown type: {fib_type}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    _display_numbers(numbers, f"{fib_type.title()} Sequence ({variant})", fmt)


@generate.command()
@click.option("--type", "seq_type", default="arithmetic",
              type=click.Choice([
                  "arithmetic", "geometric", "harmonic", "triangular", "square",
                  "pentagonal", "hexagonal", "catalan", "bell", "collatz",
                  "recaman", "padovan", "perrin", "look_and_say"
              ]),
              help="Sequence type")
@click.option("--count", "-n", default=15, type=int)
@click.option("--start", type=float, default=None, help="Starting value (for arithmetic/geometric)")
@click.option("--step", type=float, default=None, help="Step/ratio (for arithmetic/geometric)")
@click.option("--n-val", type=int, default=27, help="N value for Collatz sequence")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "csv", "plain"]))
def sequence(seq_type: str, count: int, start: Optional[float], step: Optional[float],
             n_val: int, fmt: str):
    """Generate mathematical sequences."""
    from src.core.generators.sequence_generator import SequenceGeneratorFactory

    kwargs = {}
    if seq_type == "arithmetic":
        if start is not None:
            kwargs["start"] = start
        if step is not None:
            kwargs["difference"] = step
    elif seq_type == "geometric":
        if start is not None:
            kwargs["start"] = start
        if step is not None:
            kwargs["ratio"] = step
    elif seq_type == "collatz":
        kwargs["n"] = n_val

    with console.status(f"Generating {seq_type} sequence..."):
        try:
            numbers = SequenceGeneratorFactory.generate(seq_type, count, **kwargs)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    _display_numbers(numbers, f"{seq_type.title()} Sequence", fmt)


@generate.command()
@click.option("--distribution", "-d", default="normal",
              type=click.Choice(["normal", "poisson", "exponential", "gamma", "beta", "uniform", "binomial"]),
              help="Statistical distribution")
@click.option("--count", "-n", default=100, type=int)
@click.option("--mean", "mu", type=float, default=0.0, help="Mean (normal distribution)")
@click.option("--std", "sigma", type=float, default=1.0, help="Std dev (normal distribution)")
@click.option("--rate", type=float, default=1.0, help="Rate parameter (Poisson/exponential)")
@click.option("--seed", type=int, default=None)
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "csv", "plain"]))
def stats(distribution: str, count: int, mu: float, sigma: float, rate: float,
          seed: Optional[int], fmt: str):
    """Generate numbers from statistical distributions."""
    from src.core.generators.statistical_generator import StatisticalGeneratorFactory

    params = {}
    if distribution in ("normal", "gaussian"):
        params = {"mu": mu, "sigma": sigma}
    elif distribution in ("poisson",):
        params = {"lam": rate}
    elif distribution in ("exponential",):
        params = {"rate": rate}
    elif distribution == "gamma":
        params = {"shape": mu if mu > 0 else 2.0, "rate": rate}
    elif distribution == "beta":
        params = {"alpha": mu if mu > 0 else 2.0, "beta": sigma if sigma > 0 else 2.0}
    elif distribution == "uniform":
        params = {"low": mu, "high": max(mu + 1, sigma)}
    elif distribution == "binomial":
        params = {"n": int(max(1, mu)) if mu > 0 else 10, "p": min(0.99, max(0.01, rate))}

    with console.status(f"Generating {count} {distribution} samples..."):
        try:
            gen = StatisticalGeneratorFactory.create(distribution, seed=seed, **params)
            numbers = gen.generate(count)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    _display_numbers([round(n, 4) for n in numbers],
                     f"{distribution.title()} Distribution (n={count})", fmt)


@generate.command()
@click.option("--type", "token_type", default="hex",
              type=click.Choice(["hex", "urlsafe", "numeric", "alphanumeric", "uuid"]))
@click.option("--count", "-n", default=5, type=int)
@click.option("--length", default=32, type=int, help="Token length in chars")
def crypto(token_type: str, count: int, length: int):
    """Generate cryptographically secure tokens and random numbers."""
    from src.core.generators.crypto_generator import TokenGenerator, UUIDGenerator

    if token_type == "uuid":
        tokens = UUIDGenerator.generate(count)
    else:
        gen = TokenGenerator(length)
        tokens = gen.generate_tokens(count, token_type)

    table = Table(title="Secure Tokens", show_header=True)
    table.add_column("Index", style="dim", justify="right")
    table.add_column("Token", style="cyan")
    for i, t in enumerate(tokens):
        table.add_row(str(i + 1), str(t))
    console.print(table)
