"""Output formatters for token counts."""

import csv
import json
import sys
from dataclasses import replace
from enum import StrEnum
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from toko.cost import format_cost, format_cost_value
from toko.output_format import OutputFormat
from toko.sort_order import SortOrder

if TYPE_CHECKING:
    from toko.result import Caveat, Retirement, TokenCount

# The version of the JSON document this module emits. Every run produces the same
# envelope -- results, totals, and a count object whose keys never depend on which
# flags were passed -- so a consumer writes one reader and it keeps working.
SCHEMA_VERSION = 1


class SourceKind(StrEnum):
    TEXT = "text"
    FILE = "file"
    URL = "url"


def _plain_table(*, include_header: bool) -> Table:
    # No box and no edge padding, so rows read like conventional CLI output: columns
    # separated by spaces and starting at column 0.
    return Table(
        show_header=include_header, header_style="bold", box=None, pad_edge=False
    )


def _render_table(table: Table, *, width: int | None = None) -> str:
    output = StringIO()
    # Rich reports a hardcoded 80x25 for a dumb terminal unless it is given both a width
    # and a height, which discards the width we ask for. A console that is not a terminal
    # is never dumb, so this throwaway one resolves the size rich itself would have used
    # (rich.console.Console.size) without writing anything.
    size = Console(file=output, force_terminal=False).size
    console = Console(
        file=output,
        force_terminal=True,
        width=size.width if width is None else width,
        height=size.height,
    )
    console.print(table)
    return output.getvalue().rstrip()


def format_table(
    results: dict[str, TokenCount],
    *,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format results as a table using rich."""
    table = _plain_table(include_header=include_header)
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right", style="green")

    if show_costs:
        table.add_column("Cost", justify="right", style="yellow")

    for model, counted in results.items():
        row = [model, f"{counted.count:,}"]
        if show_costs:
            row.append(format_cost(counted.cost))
        table.add_row(*row)

    return _render_table(table)


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


def _caveat_payload(caveat: Caveat) -> dict[str, object]:
    return {
        "kind": caveat.kind.value,
        "model": caveat.model,
        "message": caveat.message,
        "encoding": caveat.encoding,
        "tokenizer": caveat.tokenizer,
        "reason": caveat.reason,
    }


def _retirement_payload(retirement: Retirement | None) -> dict[str, object] | None:
    if retirement is None:
        return None
    return {
        "model": retirement.model,
        "date": retirement.date,
        "redirects_to": retirement.redirects_to,
    }


def _count_payload(model: str, counted: TokenCount) -> dict[str, object]:
    # Every key is present on every count, whatever was asked for: a reader that
    # tests for a key would otherwise be testing which flags the run was given.
    # Costs stay raw numbers (or null) rather than the display strings the table
    # formats use, so the output remains machine-readable.
    return {
        "model": model,
        "tokens": counted.count,
        "approximate": counted.approximate,
        "cost": counted.cost,
        "caveats": [_caveat_payload(caveat) for caveat in counted.caveats],
        "retirement": _retirement_payload(counted.retirement),
    }


def _counts_payload(
    results: dict[str, TokenCount], models: list[str]
) -> list[dict[str, object]]:
    # Ordered by the document's one model order rather than by whatever this
    # particular source happens to hold, so every count array in the document lists
    # its models the same way and a reader can line them up.
    return [
        _count_payload(model, results[model]) for model in models if model in results
    ]


def _source_payload(name: str) -> dict[str, object]:
    # The same test the CLI itself uses to decide whether a path is a URL, applied
    # to the display name it stored, so the two can never disagree about a source.
    kind = (
        SourceKind.URL if name.startswith(("http://", "https://")) else SourceKind.FILE
    )
    return {"kind": kind.value, "name": name}


def _envelope(results: list[dict[str, object]], totals: list[dict[str, object]]) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "results": results, "totals": totals},
        indent=2,
    )


def format_json(results: dict[str, TokenCount], *, total_only: bool = False) -> str:
    """Format counts of a text input (``--text`` or stdin) as the JSON envelope."""
    counts = _counts_payload(results, list(results))
    # One source and one count per model, so the totals repeat the results rather
    # than summing anything. Repeating them keeps the document one shape.
    sources = (
        []
        if total_only
        else [
            {"source": {"kind": SourceKind.TEXT.value, "name": None}, "counts": counts}
        ]
    )
    return _envelope(sources, counts)


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

    rows: list[list[str]] = []
    if include_header:
        header = ["model", "tokens"]
        if show_costs:
            header.append("cost")
        if show_approximate:
            header.append("approximate")
        rows.append(header)

    for model, counted in results.items():
        fields = [model, str(counted.count)]
        if show_costs:
            fields.append(format_cost_value(counted.cost))
        if show_approximate:
            fields.append(_approximate_field(counted.approximate))
        rows.append(fields)
    return _render_delimited(rows, separator)


def _approximate_field(approximate: bool) -> str:
    return "true" if approximate else "false"


def _render_delimited(rows: list[list[str]], separator: str) -> str:
    # CSV goes through csv.writer so a comma, quote, or newline in a name stays
    # inside one field. TSV has no quoting convention, so it stays a plain join.
    if separator != ",":
        return "\n".join(separator.join(row) for row in rows)
    buffer = StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().rstrip("\n")


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
    total_only: bool = False,
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
        return format_json(results, total_only=total_only)
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
) -> str:
    totals = _compute_totals(file_results, models=models)
    # --total-only drops the per-source rows and nothing else: the document keeps
    # both keys, so it parses with the same reader as a full run.
    results = (
        []
        if total_only
        else [
            {
                "source": _source_payload(filename),
                "counts": _counts_payload(counts, models),
            }
            for filename, counts in file_results.items()
        ]
    )
    return _envelope(results, _counts_payload(totals, models))


def _collect_models(
    file_results: dict[str, dict[str, TokenCount]], requested: list[str] | None = None
) -> list[str]:
    # The one order a run's output uses: the table columns, the delimited columns and
    # both JSON arrays all read it, so nothing in a document has to be re-read against a
    # second order. `requested` is the order the models were asked for, which is the
    # order to show them in whichever sources each one could be counted for. Without it
    # the best available order is first encounter, which is the same order right up
    # until a model fails on an early source and succeeds on a later one.
    counted = dict.fromkeys(
        model for counts in file_results.values() for model in counts
    )
    if requested is None:
        return list(counted)
    # A model no source could be counted for has no column and no total to put in one,
    # so it drops out; the rest keep the order they were asked in.
    return [model for model in dict.fromkeys(requested) if model in counted]


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
    rows: list[list[str]] = []
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
        rows.append(headers)

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
        rows.append(total_row)
        return _render_delimited(rows, separator)

    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            row.extend(
                _delimited_cells(
                    model_counts.get(model),
                    show_costs=show_costs,
                    show_approximate=show_approximate,
                )
            )
        rows.append(row)

    return _render_delimited(rows, separator)


def _delimited_cells(
    counted: TokenCount | None, *, show_costs: bool, show_approximate: bool
) -> list[str]:
    if counted is None:
        # No count at all for this file and model. The cost cell stays empty rather
        # than saying N/A, so the column holds only numbers and blanks.
        cells = ["N/A"]
        if show_costs:
            cells.append("")
        if show_approximate:
            cells.append("N/A")
        return cells

    cells = [str(counted.count)]
    if show_costs:
        cells.append(format_cost_value(counted.cost))
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
    table = _plain_table(include_header=include_header)
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

    return _render_table(table, width=200)


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
        # later ones would hide a failure the total is built from. Repeats collapse
        # on the whole structured record rather than on its wording, so the usual
        # column-wide identical caveat reads exactly as one file's does -- and they
        # stay separate objects, so no punctuation has to survive a round trip.
        caveats: list[Caveat] = []
        for model_counts in file_results.values():
            counted = model_counts.get(model)
            if counted is None:
                continue
            for caveat in counted.caveats:
                if caveat not in caveats:
                    caveats.append(caveat)
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
        if model in totals and totals[model].caveats != tuple(caveats):
            totals[model] = replace(totals[model], caveats=tuple(caveats))

    return totals


def _sort_file_results(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    sort_order: SortOrder | str,
) -> dict[str, dict[str, TokenCount]]:
    # Reordered once, before the per-format code runs, so every format agrees on the row
    # order — including JSON, whose keys keep their insertion order.
    if sort_order == SortOrder.INPUT:
        return file_results

    if sort_order == SortOrder.PATH:
        # The keys are the paths as the File column shows them, so a plain string sort
        # is the order a reader sees, and it keeps a directory's files together.
        return dict(sorted(file_results.items(), key=lambda item: item[0]))

    if sort_order != SortOrder.COUNT:
        raise ValueError(f"Unknown sort order: {sort_order}")

    if not models:
        return file_results
    leading = models[0]

    def rank(item: tuple[str, dict[str, TokenCount]]) -> tuple[bool, int, str]:
        filename, model_counts = item
        counted = model_counts.get(leading)
        # A file the leading model could not count has no number to rank by, so it sorts
        # after the ones that do rather than to the top on a stand-in zero. The filename
        # breaks ties, so equal counts stay in a stable, readable order.
        if counted is None:
            return (True, 0, filename)
        return (False, -counted.count, filename)

    return dict(sorted(file_results.items(), key=rank))


def format_file_table(
    file_results: dict[str, dict[str, TokenCount]],
    output_format: OutputFormat | str = "text",
    total_only: bool = False,
    *,
    models: list[str] | None = None,
    show_costs: bool = False,
    include_header: bool = True,
    sort_order: SortOrder | str = SortOrder.INPUT,
) -> str:
    """Format per-file token counts with files as rows and models as columns.

    ``models`` is the order the models were asked for; the columns and the JSON arrays
    follow it rather than the order the counts happen to be stored in. Omit it and the
    order is first encounter in ``file_results``.
    """
    models = _collect_models(file_results, models)
    if not total_only:
        # A total-only run prints no file rows to order, and sorting anyway would leak
        # into the one row it does print: _compute_totals joins the per-file caveats in
        # iteration order, so the row's caveat text would follow --sort.
        file_results = _sort_file_results(
            file_results, models=models, sort_order=sort_order
        )

    if output_format == OutputFormat.JSON:
        return _format_file_json(file_results, models=models, total_only=total_only)

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
