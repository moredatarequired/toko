"""Output formatters for token counts."""

import json
import sys
from io import StringIO
from typing import cast

from rich.console import Console
from rich.table import Table

from toko.cost import format_cost


def format_table(
    results: dict[str, int],
    *,
    costs: dict[str, float | None] | None = None,
    include_header: bool = True,
) -> str:
    """Format results as a table using rich.

    Args:
        results: Dictionary mapping model names to token counts
        costs: Optional dictionary mapping model names to costs

    Returns:
        Table-formatted output
    """
    table = Table(show_header=include_header, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right", style="green")

    if costs:
        table.add_column("Cost", justify="right", style="yellow")

    for model, count in results.items():
        if costs and model in costs:
            cost_str = format_cost(costs[model])
            table.add_row(model, f"{count:,}", cost_str)
        else:
            table.add_row(model, f"{count:,}")

    # Render to string
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    console.print(table)
    return output.getvalue().rstrip()


def format_text(
    results: dict[str, int],
    total_only: bool = False,
    *,
    costs: dict[str, float | None] | None = None,
    include_header: bool = True,
) -> str:
    """Format results as human-readable text.

    Args:
        results: Dictionary mapping model names to token counts
        total_only: If True, only show total count
        costs: Optional dictionary mapping model names to costs

    Returns:
        Formatted text output
    """
    if len(results) == 1 and not costs:
        # Single model without costs - just show the count
        count = next(iter(results.values()))
        if total_only:
            return str(count)
        return str(count)

    # Multiple models or costs requested - use table format
    return format_table(results, costs=costs, include_header=include_header)


def format_json(results: dict[str, int]) -> str:
    """Format results as JSON.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        JSON-formatted output
    """
    return json.dumps(results, indent=2)


def format_csv(
    results: dict[str, int],
    *,
    include_header: bool = True,
    costs: dict[str, float | None] | None = None,
) -> str:
    """Format results as CSV.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        CSV-formatted output
    """
    lines: list[str] = []
    if include_header:
        header = ["model", "tokens"]
        if costs:
            header.append("cost")
        lines.append(",".join(header))
    for model, count in results.items():
        fields = [model, str(count)]
        if costs:
            fields.append(format_cost(costs.get(model)))
        lines.append(",".join(fields))
    return "\n".join(lines)


def format_tsv(
    results: dict[str, int],
    *,
    include_header: bool = True,
    costs: dict[str, float | None] | None = None,
) -> str:
    """Format results as TSV.

    Args:
        results: Dictionary mapping model names to token counts

    Returns:
        TSV-formatted output
    """
    lines: list[str] = []
    if include_header:
        header = ["model", "tokens"]
        if costs:
            header.append("cost")
        lines.append("\t".join(header))
    for model, count in results.items():
        fields = [model, str(count)]
        if costs:
            fields.append(format_cost(costs.get(model)))
        lines.append("\t".join(fields))
    return "\n".join(lines)


def format_output(
    results: dict[str, int],
    output_format: str = "text",
    total_only: bool = False,
    *,
    costs: dict[str, float | None] | None = None,
    include_header: bool = True,
) -> str:
    """Format token count results according to specified format.

    Args:
        results: Dictionary mapping model names to token counts
        output_format: Output format (text, json, csv, tsv)
        total_only: If True, only show total count (text format only)
        costs: Optional dictionary mapping model names to costs

    Returns:
        Formatted output string

    Raises:
        ValueError: If format is not supported
    """
    if output_format == "text":
        return format_text(
            results, total_only, costs=costs, include_header=include_header
        )
    if output_format == "json":
        return format_json(results)
    if output_format == "csv":
        return format_csv(results, include_header=include_header, costs=costs)
    if output_format == "tsv":
        return format_tsv(results, include_header=include_header, costs=costs)
    raise ValueError(f"Unknown format: {output_format}")


def _collect_models(file_results: dict[str, dict[str, int]]) -> list[str]:
    return sorted({model for counts in file_results.values() for model in counts})


def _format_file_table_delimited(
    file_results: dict[str, dict[str, int]],
    *,
    models: list[str],
    separator: str,
    include_header: bool,
    costs: dict[str, dict[str, float | None]] | None,
) -> str:
    lines: list[str] = []
    if costs:
        headers = ["file"] + [
            label for model in models for label in (f"{model}_tokens", f"{model}_cost")
        ]
    else:
        headers = ["file", *models]

    if include_header:
        lines.append(separator.join(headers))

    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            if model in model_counts:
                row.append(str(model_counts[model]))
                if costs:
                    cost_val = costs.get(filename, {}).get(model)
                    row.append(format_cost(cost_val))
            else:
                row.append("N/A")
                if costs:
                    row.append("N/A")
        lines.append(separator.join(row))

    return "\n".join(lines)


def _format_file_table_text(
    file_results: dict[str, dict[str, int]],
    *,
    models: list[str],
    total_only: bool,
    include_header: bool,
    costs: dict[str, dict[str, float | None]] | None,
) -> str:
    table = Table(show_header=include_header, header_style="bold")
    table.add_column("File", style="cyan", no_wrap=False)

    if costs:
        for model in models:
            table.add_column(f"{model}\nTokens", justify="right", style="green")
            table.add_column(f"{model}\nCost", justify="right", style="yellow")
    else:
        for model in models:
            table.add_column(model, justify="right", style="green")

    rows, totals, total_costs = _build_table_rows(
        file_results, models=models, costs=costs
    )

    for row in rows:
        table.add_row(*row)

    if not total_only and len(file_results) > 1:
        total_row: list[str] = ["TOTAL"]
        for model in models:
            total_row.append(f"{totals[model]:,}")
            if costs and total_costs is not None:
                total_row.append(format_cost(total_costs[model]))
        table.add_row(*total_row, style="bold")

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=200)
    console.print(table)
    return output.getvalue().rstrip()


def _build_table_rows(
    file_results: dict[str, dict[str, int]],
    *,
    models: list[str],
    costs: dict[str, dict[str, float | None]] | None,
) -> tuple[list[list[str]], dict[str, int], dict[str, float] | None]:
    totals = cast("dict[str, int]", dict.fromkeys(models, 0))
    total_costs: dict[str, float] | None = (
        cast("dict[str, float]", dict.fromkeys(models, 0.0)) if costs else None
    )
    rows: list[list[str]] = []

    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            if model in model_counts:
                count = model_counts[model]
                totals[model] += count
                row.append(f"{count:,}")

                if costs:
                    cost_val = costs.get(filename, {}).get(model)
                    if total_costs is not None and cost_val is not None:
                        total_costs[model] += cost_val
                    row.append(format_cost(cost_val))
            else:
                row.append("N/A")
                if costs:
                    row.append("N/A")
        rows.append(row)

    return rows, totals, total_costs


def format_file_table(
    file_results: dict[str, dict[str, int]],
    output_format: str = "text",
    total_only: bool = False,
    *,
    costs: dict[str, dict[str, float | None]] | None = None,
    include_header: bool = True,
) -> str:
    """Format per-file token counts with files as rows and models as columns."""
    if output_format == "json":
        return json.dumps(file_results, indent=2)

    models = _collect_models(file_results)

    if output_format in ("csv", "tsv"):
        separator = "," if output_format == "csv" else "\t"
        return _format_file_table_delimited(
            file_results,
            models=models,
            separator=separator,
            include_header=include_header,
            costs=costs,
        )

    return _format_file_table_text(
        file_results,
        models=models,
        total_only=total_only,
        include_header=include_header,
        costs=costs,
    )


def is_stdin_empty() -> bool:
    """Check if stdin is empty or is a TTY.

    Returns:
        True if stdin should be considered empty (is a TTY or has no data)
    """
    return sys.stdin.isatty()
