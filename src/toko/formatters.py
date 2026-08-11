"""Output formatters for token counts."""

import json
import sys
from dataclasses import replace
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from toko.cost import format_cost
from toko.output_format import OutputFormat

if TYPE_CHECKING:
    from toko.result import TokenCount


def format_table(
    results: dict[str, TokenCount],
    *,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format results as a table using rich."""
    table = Table(show_header=include_header, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right", style="green")

    if show_costs:
        table.add_column("Cost", justify="right", style="yellow")

    for model, counted in results.items():
        row = [model, f"{counted.count:,}"]
        if show_costs:
            row.append(format_cost(counted.cost))
        table.add_row(*row)

    # Render to string
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    console.print(table)
    return output.getvalue().rstrip()


def format_text(
    results: dict[str, TokenCount],
    *,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format results as human-readable text."""
    if len(results) == 1 and not show_costs:
        # Single model without costs - just show the count
        return str(next(iter(results.values())).count)

    # Multiple models or costs requested - use table format
    return format_table(results, show_costs=show_costs, include_header=include_header)


def _any_approximate(results: dict[str, TokenCount]) -> bool:
    return any(counted.approximate for counted in results.values())


def _any_caveat(results: dict[str, TokenCount]) -> bool:
    return any(counted.caveat is not None for counted in results.values())


def _json_payload(
    results: dict[str, TokenCount],
    *,
    show_costs: bool,
    show_approximate: bool | None = None,
    any_caveat: bool | None = None,
) -> dict[str, object]:
    # Callers spanning several payloads pass the document-wide answers, so one
    # annotated count anywhere gives every file the same keys.
    if show_approximate is None:
        show_approximate = _any_approximate(results)
    if any_caveat is None:
        any_caveat = _any_caveat(results)

    if not show_costs and not show_approximate and not any_caveat:
        return {model: counted.count for model, counted in results.items()}

    # Costs stay as raw numbers (or null) rather than the display strings the
    # table formats use, so the output remains machine-readable.
    payload: dict[str, object] = {}
    for model, counted in results.items():
        entry: dict[str, object] = {"tokens": counted.count}
        if show_costs:
            entry["cost"] = counted.cost
        if show_approximate:
            # Present on every entry even when exact, so one document never mixes shapes.
            entry["approximate"] = counted.approximate
        # Independent of the approximate gate: a caveat is worth reporting on its
        # own, and its visibility should not depend on some sibling model.
        if counted.caveat is not None:
            entry["caveat"] = counted.caveat
        payload[model] = entry
    return payload


def format_json(results: dict[str, TokenCount], *, show_costs: bool = False) -> str:
    """Format results as JSON."""
    return json.dumps(_json_payload(results, show_costs=show_costs), indent=2)


def _format_delimited(
    results: dict[str, TokenCount],
    *,
    separator: str,
    include_header: bool,
    show_costs: bool,
) -> str:
    # The column only appears when it has something to say, so runs that are wholly
    # exact keep the two-column shape every existing consumer parses.
    show_approximate = _any_approximate(results)

    lines: list[str] = []
    if include_header:
        header = ["model", "tokens"]
        if show_costs:
            header.append("cost")
        if show_approximate:
            header.append("approximate")
        lines.append(separator.join(header))

    for model, counted in results.items():
        fields = [model, str(counted.count)]
        if show_costs:
            fields.append(format_cost(counted.cost))
        if show_approximate:
            fields.append(_approximate_field(counted.approximate))
        lines.append(separator.join(fields))
    return "\n".join(lines)


def _approximate_field(approximate: bool) -> str:
    return "true" if approximate else "false"


def format_csv(
    results: dict[str, TokenCount],
    *,
    include_header: bool = True,
    show_costs: bool = False,
) -> str:
    """Format results as CSV."""
    return _format_delimited(
        results, separator=",", include_header=include_header, show_costs=show_costs
    )


def format_tsv(
    results: dict[str, TokenCount],
    *,
    include_header: bool = True,
    show_costs: bool = False,
) -> str:
    """Format results as TSV."""
    return _format_delimited(
        results, separator="\t", include_header=include_header, show_costs=show_costs
    )


def format_output(
    results: dict[str, TokenCount],
    output_format: OutputFormat | str = "text",
    *,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format token count results according to specified format.

    Raises:
        ValueError: If format is not supported
    """
    if output_format == OutputFormat.TEXT:
        return format_text(
            results, show_costs=show_costs, include_header=include_header
        )
    if output_format == OutputFormat.JSON:
        return format_json(results, show_costs=show_costs)
    if output_format == OutputFormat.CSV:
        return format_csv(results, include_header=include_header, show_costs=show_costs)
    if output_format == OutputFormat.TSV:
        return format_tsv(results, include_header=include_header, show_costs=show_costs)
    raise ValueError(f"Unknown format: {output_format}")


def _format_file_json(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    total_only: bool,
    show_costs: bool,
) -> str:
    if total_only:
        totals = _compute_totals(file_results, models=models)
        return json.dumps(_json_payload(totals, show_costs=show_costs), indent=2)

    # One annotated count anywhere switches every file to the object form, so a
    # reader never has to test which shape a given entry took.
    approximate_anywhere = any(
        _any_approximate(counts) for counts in file_results.values()
    )
    caveat_anywhere = any(_any_caveat(counts) for counts in file_results.values())
    payload = {
        filename: _json_payload(
            counts,
            show_costs=show_costs,
            show_approximate=approximate_anywhere,
            any_caveat=caveat_anywhere,
        )
        for filename, counts in file_results.items()
    }
    return json.dumps(payload, indent=2)


def _collect_models(file_results: dict[str, dict[str, TokenCount]]) -> list[str]:
    return sorted({model for counts in file_results.values() for model in counts})


def _format_file_table_delimited(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    separator: str,
    total_only: bool,
    include_header: bool,
    show_costs: bool,
) -> str:
    show_approximate = any(_any_approximate(counts) for counts in file_results.values())
    lines: list[str] = []
    if show_costs or show_approximate:
        headers = ["file"]
        for model in models:
            headers.append(f"{model}_tokens")
            if show_costs:
                headers.append(f"{model}_cost")
            if show_approximate:
                headers.append(f"{model}_approximate")
    else:
        headers = ["file", *models]

    if include_header:
        lines.append(separator.join(headers))

    if total_only:
        totals = _compute_totals(file_results, models=models)
        total_row: list[str] = ["TOTAL"]
        for model in models:
            total_row.extend(
                _delimited_cells(
                    totals[model],
                    show_costs=show_costs,
                    show_approximate=show_approximate,
                )
            )
        lines.append(separator.join(total_row))
        return "\n".join(lines)

    per_model_columns = 1 + int(show_costs) + int(show_approximate)
    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            counted = model_counts.get(model)
            if counted is None:
                row.extend(["N/A"] * per_model_columns)
            else:
                row.extend(
                    _delimited_cells(
                        counted,
                        show_costs=show_costs,
                        show_approximate=show_approximate,
                    )
                )
        lines.append(separator.join(row))

    return "\n".join(lines)


def _delimited_cells(
    counted: TokenCount, *, show_costs: bool, show_approximate: bool
) -> list[str]:
    cells = [str(counted.count)]
    if show_costs:
        cells.append(format_cost(counted.cost))
    if show_approximate:
        cells.append(_approximate_field(counted.approximate))
    return cells


def _format_file_table_text(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    total_only: bool,
    include_header: bool,
    show_costs: bool,
) -> str:
    table = Table(show_header=include_header, header_style="bold")
    table.add_column("File", style="cyan", no_wrap=False)

    if show_costs:
        for model in models:
            table.add_column(f"{model}\nTokens", justify="right", style="green")
            table.add_column(f"{model}\nCost", justify="right", style="yellow")
    else:
        for model in models:
            table.add_column(model, justify="right", style="green")

    if not total_only:
        for row in _build_table_rows(
            file_results, models=models, show_costs=show_costs
        ):
            table.add_row(*row)

    if total_only or len(file_results) > 1:
        totals = _compute_totals(file_results, models=models)
        total_row: list[str] = ["TOTAL"]
        for model in models:
            total_row.append(f"{totals[model].count:,}")
            if show_costs:
                total_row.append(format_cost(totals[model].cost))
        table.add_row(*total_row, style="bold")

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=200)
    console.print(table)
    return output.getvalue().rstrip()


def _build_table_rows(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    show_costs: bool,
) -> list[list[str]]:
    rows: list[list[str]] = []

    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            counted = model_counts.get(model)
            if counted is None:
                row.append("N/A")
                if show_costs:
                    row.append("N/A")
            else:
                row.append(f"{counted.count:,}")
                if show_costs:
                    row.append(format_cost(counted.cost))
        rows.append(row)

    return rows


def _sum_costs(running: float | None, addition: float | None) -> float | None:
    # Seeded with None, not 0.0: a model no file could be priced for has no
    # total, and reporting $0.000000 for it reads as a confident free. A model
    # only some files could be priced for keeps the sum of those files.
    if addition is None:
        return running
    return addition if running is None else running + addition


def _compute_totals(
    file_results: dict[str, dict[str, TokenCount]], *, models: list[str]
) -> dict[str, TokenCount]:
    totals: dict[str, TokenCount] = {}

    for model in models:
        # Distinct caveats in one column all survive, in encounter order: files can
        # fail differently (the xAI caveat names the error it saw), and dropping the
        # later ones would hide a failure the total is built from. Repeats collapse,
        # so the usual column-wide identical caveat reads exactly as one file's does.
        caveats: list[str] = []
        for model_counts in file_results.values():
            counted = model_counts.get(model)
            if counted is None:
                continue
            if counted.caveat is not None and counted.caveat not in caveats:
                caveats.append(counted.caveat)
            running = totals.get(model)
            if running is None:
                totals[model] = counted
                continue
            totals[model] = replace(
                running,
                count=running.count + counted.count,
                cost=_sum_costs(running.cost, counted.cost),
                approximate=running.approximate or counted.approximate,
            )
        merged_caveat = "; ".join(caveats) if caveats else None
        if model in totals and totals[model].caveat != merged_caveat:
            totals[model] = replace(totals[model], caveat=merged_caveat)

    return totals


def format_file_table(
    file_results: dict[str, dict[str, TokenCount]],
    output_format: OutputFormat | str = "text",
    total_only: bool = False,
    *,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format per-file token counts with files as rows and models as columns."""
    models = _collect_models(file_results)

    if output_format == OutputFormat.JSON:
        return _format_file_json(
            file_results, models=models, total_only=total_only, show_costs=show_costs
        )

    if output_format in (OutputFormat.CSV, OutputFormat.TSV):
        return _format_file_table_delimited(
            file_results,
            models=models,
            separator="," if output_format == OutputFormat.CSV else "\t",
            total_only=total_only,
            include_header=include_header,
            show_costs=show_costs,
        )

    if output_format == OutputFormat.TEXT:
        return _format_file_table_text(
            file_results,
            models=models,
            total_only=total_only,
            include_header=include_header,
            show_costs=show_costs,
        )

    raise ValueError(f"Unknown format: {output_format}")


def is_stdin_empty() -> bool:
    """Check if stdin is empty or is a TTY.

    Returns:
        True if stdin should be considered empty (is a TTY or has no data)
    """
    return sys.stdin.isatty()
