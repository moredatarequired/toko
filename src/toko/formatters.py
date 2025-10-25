"""Output formatters for token counts."""

import json
import sys
from io import StringIO

from rich.console import Console
from rich.table import Table


def format_table(results: dict[str, int]) -> str:
    """Format results as a table using rich.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        Table-formatted output
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right", style="green")

    for model, count in results.items():
        table.add_row(model, f"{count:,}")

    # Render to string
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    console.print(table)
    return output.getvalue().rstrip()


def format_text(results: dict[str, int], total_only: bool = False) -> str:
    """Format results as human-readable text.

    Args:
        results: Dictionary mapping model names to token counts
        total_only: If True, only show total count

    Returns:
        Formatted text output
    """
    if len(results) == 1:
        # Single model - just show the count
        count = next(iter(results.values()))
        if total_only:
            return str(count)
        return f"{count:,} tokens"

    # Multiple models - use table format
    return format_table(results)


def format_json(results: dict[str, int]) -> str:
    """Format results as JSON.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        JSON-formatted output
    """
    return json.dumps(results, indent=2)


def format_csv(results: dict[str, int]) -> str:
    """Format results as CSV.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        CSV-formatted output
    """
    lines = ["model,tokens"]
    for model, count in results.items():
        lines.append(f"{model},{count}")
    return "\n".join(lines)


def format_tsv(results: dict[str, int]) -> str:
    """Format results as TSV.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        TSV-formatted output
    """
    lines = ["model\ttokens"]
    for model, count in results.items():
        lines.append(f"{model}\t{count}")
    return "\n".join(lines)


def format_output(
    results: dict[str, int],
    output_format: str = "text",
    total_only: bool = False,
) -> str:
    """Format token count results according to specified format.

    Args:
        results: Dictionary mapping model names to token counts
        output_format: Output format (text, json, csv, tsv)
        total_only: If True, only show total count (text format only)

    Returns:
        Formatted output string

    Raises:
        ValueError: If format is not supported
    """
    if output_format == "text":
        return format_text(results, total_only)
    if output_format == "json":
        return format_json(results)
    if output_format == "csv":
        return format_csv(results)
    if output_format == "tsv":
        return format_tsv(results)
    raise ValueError(f"Unknown format: {output_format}")


def is_stdin_empty() -> bool:
    """Check if stdin is empty or is a TTY.

    Returns:
        True if stdin should be considered empty (is a TTY or has no data)
    """
    return sys.stdin.isatty()
