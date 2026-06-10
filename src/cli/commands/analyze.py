"""
CLI analyze subcommands for statistical analysis of number sequences.
"""

import json
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def _read_data(input_file: Optional[str], data_str: Optional[str]) -> List[float]:
    """Read data from file or string."""
    if input_file:
        try:
            with open(input_file, "r") as f:
                content = f.read().strip()
            # Try JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [float(x) for x in data]
            except json.JSONDecodeError:
                pass
            # Try CSV
            data = [float(x.strip()) for x in content.replace("\n", ",").split(",") if x.strip()]
            return data
        except FileNotFoundError:
            console.print(f"[red]File not found: {input_file}[/red]")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]Invalid data in file: {e}[/red]")
            sys.exit(1)
    elif data_str:
        try:
            return [float(x.strip()) for x in data_str.split(",") if x.strip()]
        except ValueError as e:
            console.print(f"[red]Invalid data: {e}[/red]")
            sys.exit(1)
    else:
        console.print("[red]Must provide --input file or --data string[/red]")
        sys.exit(1)


@click.group()
def analyze():
    """Analyze number sequences statistically."""
    pass


@analyze.command()
@click.option("--input", "-i", "input_file", type=str, default=None, help="Input CSV/JSON file")
@click.option("--data", "-d", type=str, default=None, help="Comma-separated numbers")
@click.option("--tests", default="all",
              type=click.Choice(["all", "basic", "advanced"]),
              help="Which tests to run")
def statistical(input_file: Optional[str], data: Optional[str], tests: str):
    """Run statistical analysis on a number sequence."""
    from src.core.analyzers.statistical_analyzer import StatisticalAnalyzer

    numbers = _read_data(input_file, data)
    if len(numbers) < 2:
        console.print("[red]Need at least 2 data points[/red]")
        sys.exit(1)

    analyzer = StatisticalAnalyzer(numbers)
    desc = analyzer.describe()

    # Basic stats table
    table = Table(title="Statistical Analysis", show_header=True)
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green", justify="right")

    basic_stats = [
        ("Count", desc["count"]),
        ("Mean", f"{desc['mean']:.4f}"),
        ("Median", f"{desc['median']:.4f}"),
        ("Std Dev", f"{desc['std_dev']:.4f}"),
        ("Variance", f"{desc['variance']:.4f}"),
        ("Min", f"{desc['min']}"),
        ("Max", f"{desc['max']}"),
        ("Range", f"{desc['range']:.4f}"),
        ("Q1", f"{desc['q1']:.4f}"),
        ("Q3", f"{desc['q3']:.4f}"),
        ("IQR", f"{desc['iqr']:.4f}"),
    ]

    if tests in ("all", "advanced"):
        if desc["skewness"] is not None:
            basic_stats.append(("Skewness", f"{desc['skewness']:.4f}"))
        if desc["kurtosis"] is not None:
            basic_stats.append(("Kurtosis", f"{desc['kurtosis']:.4f}"))
        basic_stats.append(("Entropy", f"{desc['entropy']:.4f}"))

    for name, value in basic_stats:
        table.add_row(name, str(value))

    console.print(table)

    if tests in ("all", "advanced") and len(numbers) >= 3:
        # Chi-square test
        try:
            chi2, p = analyzer.chi_square_test()
            console.print(f"\n[bold]Chi-Square Test:[/bold] stat={chi2:.4f}, p={p:.4f} "
                          f"({'[green]pass[/green]' if p > 0.05 else '[red]fail[/red]'})")
        except Exception:
            pass

        # t-test
        try:
            t, p = analyzer.t_test(0)
            console.print(f"[bold]One-sample t-test (mu=0):[/bold] t={t:.4f}, p={p:.4f}")
        except Exception:
            pass

        # Autocorrelation
        try:
            ac = analyzer.autocorrelation(1)
            console.print(f"[bold]Autocorrelation (lag=1):[/bold] {ac:.4f}")
        except Exception:
            pass


@analyze.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
@click.option("--test", "test_name", default="all",
              type=click.Choice(["all", "frequency", "runs", "longest_run", "serial", "poker",
                                  "autocorrelation", "uniformity"]))
def randomness(input_file: Optional[str], data: Optional[str], test_name: str):
    """Test randomness properties of a number sequence."""
    from src.core.analyzers.randomness_tester import RandomnessTester

    numbers = _read_data(input_file, data)
    tester = RandomnessTester()

    with console.status("Running randomness tests..."):
        if test_name == "all":
            report = tester.run_all(numbers)
        else:
            result = tester.run_test(test_name, numbers)
            from src.core.analyzers.randomness_tester import TestReport
            report = TestReport()
            report.add_result(result)

    table = Table(title="Randomness Test Results", show_header=True)
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="bold")
    table.add_column("p-value", justify="right")
    table.add_column("Statistic", justify="right")

    for result in report.results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        table.add_row(
            result.name,
            status,
            f"{result.p_value:.4f}",
            f"{result.statistic:.4f}"
        )

    console.print(table)
    console.print(f"\n[bold]Pass Rate:[/bold] {report.passed_tests}/{report.total_tests} "
                  f"({report.pass_rate:.1%})")


@analyze.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
def patterns(input_file: Optional[str], data: Optional[str]):
    """Detect patterns in a number sequence."""
    from src.core.analyzers.pattern_detector import PatternDetector

    numbers = _read_data(input_file, data)

    with console.status("Analyzing patterns..."):
        detector = PatternDetector()
        report = detector.analyze(numbers)

    table = Table(title="Pattern Detection Results", show_header=True)
    table.add_column("Pattern Type", style="cyan")
    table.add_column("Detected", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Description")

    for p in report.patterns:
        detected = "[green]YES[/green]" if p.detected else "[red]NO[/red]"
        table.add_row(
            p.pattern_type,
            detected,
            f"{p.confidence:.2f}",
            p.description[:60]
        )

    console.print(table)

    dominant = report.dominant_pattern()
    if dominant:
        console.print(f"\n[bold green]Dominant Pattern:[/bold green] {dominant.pattern_type} "
                      f"(confidence={dominant.confidence:.2f})")


@analyze.command()
@click.option("--input", "-i", "input_file", type=str, default=None)
@click.option("--data", "-d", type=str, default=None)
@click.option("--distributions", "-D", default="all",
              type=click.Choice(["all", "normal", "exponential", "uniform", "log_normal"]))
def fit(input_file: Optional[str], data: Optional[str], distributions: str):
    """Fit statistical distributions to data."""
    from src.core.analyzers.distribution_fitter import DistributionFitter

    numbers = _read_data(input_file, data)

    with console.status("Fitting distributions..."):
        fitter = DistributionFitter()
        if distributions == "all":
            results = fitter.fit_all(numbers)
        else:
            results = [fitter.fit(numbers, distributions)]

    table = Table(title="Distribution Fit Results", show_header=True)
    table.add_column("Distribution", style="cyan")
    table.add_column("Good Fit", style="bold")
    table.add_column("KS Stat", justify="right")
    table.add_column("KS p-value", justify="right")
    table.add_column("AIC", justify="right")
    table.add_column("Parameters")

    for r in results:
        fit_str = "[green]YES[/green]" if r.is_good_fit else "[red]NO[/red]"
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in r.parameters.items())
        table.add_row(
            r.distribution,
            fit_str,
            f"{r.ks_statistic:.4f}",
            f"{r.ks_p_value:.4f}",
            f"{r.aic:.2f}",
            params_str[:50]
        )

    console.print(table)
