"""
Main CLI entry point using Click framework.
Provides the 'numgen' command with subcommands for generate, analyze, transform, and export.
"""

import sys
import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option("1.0.0", prog_name="numgen")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """
    Number Generator - A comprehensive CLI for generating, analyzing, and transforming numbers.

    Examples:
      numgen generate random --algorithm lcg --count 100
      numgen generate prime --count 50
      numgen generate fibonacci --count 30
      numgen analyze --input data.csv
      numgen transform --input data.csv --operation normalize
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Import and register subcommand groups
from src.cli.commands.generate import generate
from src.cli.commands.analyze import analyze
from src.cli.commands.transform import transform
from src.cli.commands.export import export
from src.cli.commands.plugins import plugins

cli.add_command(generate)
cli.add_command(plugins)
cli.add_command(analyze)
cli.add_command(transform)
cli.add_command(export)


if __name__ == "__main__":
    cli()
