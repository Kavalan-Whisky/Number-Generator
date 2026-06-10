"""
CLI export subcommands - export generated sequences to various formats.
"""

import csv
import json
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional

import click
from rich.console import Console

console = Console()


@click.group()
def export():
    """Export number sequences to various file formats."""
    pass


@export.command()
@click.option("--data", "-d", type=str, required=True, help="Comma-separated numbers to export")
@click.option("--format", "fmt", required=True,
              type=click.Choice(["csv", "json", "xml", "sqlite"]),
              help="Export format")
@click.option("--output", "-o", type=str, required=True, help="Output file path")
@click.option("--name", default="sequence", help="Sequence/table name")
@click.option("--metadata", default="{}", help="JSON metadata string")
def save(data: str, fmt: str, output: str, name: str, metadata: str):
    """Save a number sequence to a file."""
    try:
        numbers = [float(x.strip()) for x in data.split(",") if x.strip()]
    except ValueError as e:
        console.print(f"[red]Invalid data: {e}[/red]")
        sys.exit(1)

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    try:
        if fmt == "csv":
            _export_csv(numbers, output, name, meta)
        elif fmt == "json":
            _export_json(numbers, output, name, meta)
        elif fmt == "xml":
            _export_xml(numbers, output, name, meta)
        elif fmt == "sqlite":
            _export_sqlite(numbers, output, name, meta)
        console.print(f"[green]Exported {len(numbers)} values to {output} ({fmt})[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        sys.exit(1)


def _export_csv(numbers: List[float], path: str, name: str, meta: dict):
    """Export to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value"])
        for i, n in enumerate(numbers):
            writer.writerow([i + 1, n])


def _export_json(numbers: List[float], path: str, name: str, meta: dict):
    """Export to JSON."""
    data = {
        "name": name,
        "count": len(numbers),
        "values": numbers,
        "metadata": meta,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _export_xml(numbers: List[float], path: str, name: str, meta: dict):
    """Export to XML."""
    root = ET.Element("sequence", attrib={"name": name, "count": str(len(numbers))})
    meta_elem = ET.SubElement(root, "metadata")
    for k, v in meta.items():
        ET.SubElement(meta_elem, k).text = str(v)
    values_elem = ET.SubElement(root, "values")
    for i, n in enumerate(numbers):
        elem = ET.SubElement(values_elem, "value", attrib={"index": str(i + 1)})
        elem.text = str(n)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)


def _export_sqlite(numbers: List[float], path: str, name: str, meta: dict):
    """Export to SQLite database."""
    import sqlite3
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create table
    table = name.replace("-", "_").replace(" ", "_")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idx INTEGER NOT NULL,
            value REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert values
    cursor.executemany(
        f"INSERT INTO {table} (idx, value) VALUES (?, ?)",
        [(i + 1, n) for i, n in enumerate(numbers)]
    )

    conn.commit()
    conn.close()


@export.command()
@click.argument("input_file")
@click.option("--format", "fmt", default="json",
              type=click.Choice(["json", "csv"]))
def convert(input_file: str, fmt: str):
    """Convert between file formats."""
    try:
        with open(input_file, "r") as f:
            content = f.read().strip()
    except FileNotFoundError:
        console.print(f"[red]File not found: {input_file}[/red]")
        sys.exit(1)

    # Auto-detect input format
    try:
        data = json.loads(content)
        numbers = data.get("values", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        numbers = [float(x.strip()) for x in content.replace("\n", ",").split(",") if x.strip()]

    if fmt == "json":
        output_path = input_file.rsplit(".", 1)[0] + ".json"
        with open(output_path, "w") as f:
            json.dump({"values": numbers}, f, indent=2)
    elif fmt == "csv":
        output_path = input_file.rsplit(".", 1)[0] + ".csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "value"])
            for i, n in enumerate(numbers):
                writer.writerow([i + 1, n])

    console.print(f"[green]Converted to {output_path}[/green]")
