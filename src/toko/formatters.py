"""Output formatters for token counts."""

import json
import sys
from io import StringIO

from rich.console import Console
from rich.table import Table

from toko.cost import format_cost


def format_table(
    results: dict[str, int], *, costs: dict[str, float | None] | None = None
) -> str:
    """Format results as a table using rich.

    Args:
        results: Dictionary mapping model names to token counts
        costs: Optional dictionary mapping model names to costs

    Returns:
        Table-formatted output
    """
    table = Table(show_header=True, header_style="bold")
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
        return f"{count:,} tokens"

    # Multiple models or costs requested - use table format
    return format_table(results, costs=costs)


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
    *,
    costs: dict[str, float | None] | None = None,
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
        return format_text(results, total_only, costs=costs)
    if output_format == "json":
        return format_json(results)
    if output_format == "csv":
        return format_csv(results)
    if output_format == "tsv":
        return format_tsv(results)
    raise ValueError(f"Unknown format: {output_format}")


def format_file_table(
    file_results: dict[str, dict[str, int]],
    output_format: str = "text",
    total_only: bool = False,
    *,
    costs: dict[str, dict[str, float | None]] | None = None,
) -> str:
    """Format per-file token counts with files as rows and models as columns.

    Args:
        file_results: Dictionary mapping filenames to {model -> count}
        output_format: Output format (text, json, csv, tsv)
        total_only: If True, show totals row only
        costs: Optional dictionary mapping filenames to {model -> cost}

    Returns:
        Formatted output string
    """
    if output_format == "json":
        return json.dumps(file_results, indent=2)

    if output_format in ("csv", "tsv"):
        sep = "," if output_format == "csv" else "\t"
        # Get all unique models across all files
        all_models = sorted(
            {model for file_counts in file_results.values() for model in file_counts}
        )

        if costs:
            # With costs, create columns like: model1_tokens, model1_cost, model2_tokens, model2_cost
            headers = ["file"] + [
                item
                for model in all_models
                for item in [f"{model}_tokens", f"{model}_cost"]
            ]
        else:
            headers = ["file", *all_models]

        lines = [sep.join(headers)]

        for filename, model_counts in file_results.items():
            row = [filename]
            for model in all_models:
                if model in model_counts:
                    row.append(str(model_counts[model]))
                    if costs and filename in costs and model in costs[filename]:
                        row.append(format_cost(costs[filename][model]))
                else:
                    row.append("N/A")
                    if costs:
                        row.append("N/A")
            lines.append(sep.join(row))

        return "\n".join(lines)

    # Text format with rich table
    # Get all unique models
    all_models = sorted(
        {model for file_counts in file_results.values() for model in file_counts}
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("File", style="cyan", no_wrap=False)

    # Add columns for each model
    if costs:
        # Multi-level: each model gets tokens + cost columns
        for model in all_models:
            table.add_column(f"{model}\nTokens", justify="right", style="green")
            table.add_column(f"{model}\nCost", justify="right", style="yellow")
    else:
        for model in all_models:
            table.add_column(model, justify="right", style="green")

    # Add rows for each file
    totals = dict.fromkeys(all_models, 0)
    total_costs = dict.fromkeys(all_models, 0.0) if costs else None

    for filename, model_counts in file_results.items():
        row = [filename]
        for model in all_models:
            if model in model_counts:
                count = model_counts[model]
                totals[model] += count
                row.append(f"{count:,}")

                if costs and filename in costs and model in costs[filename]:
                    cost_val = costs[filename][model]
                    if total_costs is not None and cost_val is not None:
                        total_costs[model] += cost_val
                    row.append(format_cost(cost_val))
            else:
                row.append("N/A")
                if costs:
                    row.append("N/A")

        table.add_row(*row)

    # Add totals row if not total_only
    if not total_only and len(file_results) > 1:
        total_row = ["TOTAL"]
        for model in all_models:
            total_row.append(f"{totals[model]:,}")
            if costs and total_costs:
                total_row.append(format_cost(total_costs[model]))
        table.add_row(*total_row, style="bold")

    # Render to string
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=200)
    console.print(table)
    return output.getvalue().rstrip()


def is_stdin_empty() -> bool:
    """Check if stdin is empty or is a TTY.

    Returns:
        True if stdin should be considered empty (is a TTY or has no data)
    """
    return sys.stdin.isatty()
