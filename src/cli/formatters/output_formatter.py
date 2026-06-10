"""
Output formatting utilities for the CLI.
Provides rich, colored, tabular output.
"""

import json
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

console = Console()


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled header."""
    text = Text(title, style="bold blue")
    if subtitle:
        content = f"{title}\n[dim]{subtitle}[/dim]"
    else:
        content = title
    console.print(Panel(content, style="blue"))


def print_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold cyan]ℹ[/bold cyan] {message}")


def numbers_table(
    data: List[Union[int, float]],
    title: str = "Numbers",
    include_stats: bool = False,
) -> Table:
    """Create a rich table for displaying numbers."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", justify="right", width=8)
    table.add_column("Value", style="green", justify="right")

    if include_stats:
        table.add_column("% of Max", style="blue", justify="right")
        max_val = max(abs(x) for x in data) if data else 1
        for i, n in enumerate(data):
            pct = f"{abs(n) / max_val * 100:.1f}%" if max_val else "0%"
            table.add_row(str(i + 1), str(n), pct)
    else:
        for i, n in enumerate(data):
            table.add_row(str(i + 1), str(n))

    return table


def stats_table(stats: Dict[str, Any], title: str = "Statistics") -> Table:
    """Create a statistics table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in stats.items():
        if value is None:
            formatted = "N/A"
        elif isinstance(value, float):
            formatted = f"{value:.6f}"
        elif isinstance(value, list):
            formatted = str(value[:3]) + ("..." if len(value) > 3 else "")
        else:
            formatted = str(value)
        table.add_row(key.replace("_", " ").title(), formatted)

    return table


def comparison_table(
    data: Dict[str, List[float]],
    title: str = "Comparison",
) -> Table:
    """Create a comparison table for multiple sequences."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", justify="right")
    for name in data.keys():
        table.add_column(name, justify="right")

    max_len = max(len(v) for v in data.values()) if data else 0
    for i in range(max_len):
        row = [str(i + 1)]
        for values in data.values():
            if i < len(values):
                v = values[i]
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            else:
                row.append("-")
        table.add_row(*row)

    return table


def format_number(n: Union[int, float], style: str = "default") -> str:
    """Format a number in the given style."""
    if style == "scientific":
        return f"{n:.6e}"
    elif style == "percentage":
        return f"{n * 100:.2f}%"
    elif style == "commas":
        return f"{n:,}"
    elif style == "hex":
        return hex(int(n))
    elif style == "binary":
        return bin(int(n))
    else:
        return str(n)


def progress_context(description: str = "Processing", total: Optional[int] = None):
    """Return a rich progress context manager."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn() if total else TextColumn(""),
        TaskProgressColumn() if total else TextColumn(""),
        TimeRemainingColumn() if total else TextColumn(""),
        console=console,
    )


def print_json(data: Any, indent: int = 2) -> None:
    """Pretty print JSON data."""
    from rich.syntax import Syntax
    json_str = json.dumps(data, indent=indent, default=str)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
    console.print(syntax)


def horizontal_histogram(
    data: List[float],
    bins: int = 10,
    width: int = 40,
    title: str = "Histogram",
) -> None:
    """Print a horizontal ASCII histogram."""
    if not data:
        return

    min_v = min(data)
    max_v = max(data)
    if min_v == max_v:
        console.print(f"[yellow]All values equal: {min_v}[/yellow]")
        return

    bin_width = (max_v - min_v) / bins
    counts = [0] * bins
    for x in data:
        idx = min(int((x - min_v) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    console.print(f"\n[bold]{title}[/bold] (n={len(data)})\n")
    for i in range(bins):
        lo = min_v + i * bin_width
        hi = lo + bin_width
        bar_len = int(counts[i] / max_count * width) if max_count else 0
        bar = "█" * bar_len
        console.print(f"[{lo:8.2f},{hi:8.2f}) |{bar:<{width}} {counts[i]}")
