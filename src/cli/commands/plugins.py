"""
CLI plugins subcommands for the generator registry.
"""

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def plugins():
    """Inspect and run registered generator plugins."""
    pass


@plugins.command("list")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json"]))
def list_plugins(category, fmt):
    """List all registered generators with metadata."""
    from src.core.plugins import list_generators

    gens = list_generators(category)
    if fmt == "json":
        click.echo(json.dumps(gens, indent=2))
        return

    table = Table(title=f"Registered Generators ({len(gens)})", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="green")
    for g in gens:
        table.add_row(g["name"], g["category"], g["description"])
    console.print(table)


@plugins.command("run")
@click.argument("name")
@click.option("--count", "-n", default=10, type=int)
@click.option("--format", "fmt", default="plain",
              type=click.Choice(["json", "csv", "plain"]))
def run_plugin(name, count, fmt):
    """Run a registered generator by name."""
    from src.core.plugins import discover_builtin_generators

    registry = discover_builtin_generators()
    try:
        values = registry.generate(name, count)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps(values))
    elif fmt == "csv":
        click.echo(",".join(str(v) for v in values))
    else:
        click.echo("\n".join(str(v) for v in values))
