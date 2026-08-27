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

from toko.cost import format_cost, format_cost_value, normalize_cost
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
    models: list[str] | None = None,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format results as a table using rich."""
    models = list(results) if models is None else models
    table = _plain_table(include_header=include_header)
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right", style="green")

    if show_costs:
        table.add_column("Cost", justify="right", style="yellow")

    for model in models:
        counted = results.get(model)
        # A model that could not be counted keeps its row and says N/A, the same way
        # the per-file table already says it: a row missing outright would leave the
        # reader to notice that a model they named is not on the page. Fenced by
        # test_the_model_table_keeps_a_row_for_a_model_that_could_not_be_counted, by
        # test_the_model_table_says_na_in_the_cost_column_of_a_model_it_could_not_count
        # and, through a real terminal, by
        # test_the_terminal_table_keeps_a_row_for_a_model_it_could_not_count.
        if counted is None:
            row = [model, "N/A"]
            if show_costs:
                row.append("N/A")
        else:
            row = [model, f"{counted.count:,}"]
            if show_costs:
                row.append(format_cost(counted.cost))
        table.add_row(*row)

    return _render_table(table)


def format_text(
    results: dict[str, TokenCount],
    *,
    models: list[str] | None = None,
    show_costs: bool = False,
    include_header: bool = True,
) -> str:
    """Format results as human-readable text."""
    models = list(results) if models is None else models
    if len(models) == 1 and not show_costs:
        # One model asked for is one number printed, and an empty line when that model
        # failed: how many lines this prints is a function of the command, not of what
        # the counting produced.
        counted = results.get(models[0])
        return "" if counted is None else str(counted.count)

    # Multiple models or costs requested - use table format
    return format_table(
        results, models=models, show_costs=show_costs, include_header=include_header
    )


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


# What a count with no recorded reason says instead of saying nothing. `tokens` is
# null exactly when `reason` says why, and that has to hold by construction rather
# than by every caller remembering to pass a matching `errors` -- a library caller
# who passes `models=` for a model it has no count and no error for still gets a
# document the documented invariant is true of. Fenced by
# test_a_count_with_no_error_recorded_still_says_why_it_has_none.
NO_REASON_RECORDED = "No count was recorded for this model, and no reason was given."


def _count_payload(
    model: str, counted: TokenCount | None, reason: str | None
) -> dict[str, object]:
    # Every key is present on every count, whatever was asked for and whatever the
    # counting produced: a reader that tests for a key would otherwise be testing
    # which flags the run was given. Costs stay raw numbers (or null) rather than the
    # display strings the table formats use, so the output remains machine-readable.
    if counted is None:
        # There is no count, so there is nothing to say about it: `tokens` is null
        # exactly when `reason` says why, and the fields that describe a count say
        # nothing rather than guessing at a default.
        return {
            "model": model,
            "tokens": None,
            "approximate": None,
            "cost": None,
            "caveats": [],
            "retirement": None,
            "reason": NO_REASON_RECORDED if reason is None else reason,
        }
    return {
        "model": model,
        "tokens": counted.count,
        "approximate": counted.approximate,
        "cost": counted.cost,
        "caveats": [_caveat_payload(caveat) for caveat in counted.caveats],
        "retirement": _retirement_payload(counted.retirement),
        "reason": None,
    }


def _counts_payload(
    results: dict[str, TokenCount],
    models: list[str],
    errors: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    # One entry per model the run asked for, in the document's one model order, so
    # every count array in the document is the same length and lists its models the
    # same way. A model this source could not be counted for is present with a null
    # `tokens` and the `reason` it failed rather than absent, because which models a
    # document describes is a function of the command, not of what the counting
    # managed to produce.
    errors = errors or {}
    return [
        _count_payload(model, results.get(model), errors.get(model)) for model in models
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


def format_json(
    results: dict[str, TokenCount],
    *,
    models: list[str] | None = None,
    errors: dict[str, str] | None = None,
    total_only: bool = False,
) -> str:
    """Format counts of a text input (``--text`` or stdin) as the JSON envelope."""
    counts = _counts_payload(
        results, list(results) if models is None else models, errors
    )
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
    models: list[str],
    separator: str,
    include_header: bool,
    show_costs: bool,
) -> str:
    # Which columns there are, and how many rows follow, is decided by the command
    # alone: the models it named, and whether it asked for costs. Nothing here reads
    # the counts to choose a shape, so two runs of one command always parse alike.
    rows: list[list[str]] = []
    if include_header:
        header = ["model", "tokens"]
        if show_costs:
            header.append("cost")
        header.append("approximate")
        rows.append(header)

    rows.extend(
        [model, *_delimited_cells(results.get(model), show_costs=show_costs)]
        for model in models
    )
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
    models: list[str] | None = None,
    include_header: bool = True,
    show_costs: bool = False,
) -> str:
    """Format results as CSV."""
    return _format_delimited(
        results,
        models=list(results) if models is None else models,
        separator=",",
        include_header=include_header,
        show_costs=show_costs,
    )


def format_tsv(
    results: dict[str, TokenCount],
    *,
    models: list[str] | None = None,
    include_header: bool = True,
    show_costs: bool = False,
) -> str:
    """Format results as TSV."""
    return _format_delimited(
        results,
        models=list(results) if models is None else models,
        separator="\t",
        include_header=include_header,
        show_costs=show_costs,
    )


def format_output(
    results: dict[str, TokenCount],
    output_format: OutputFormat | str = "text",
    *,
    models: list[str] | None = None,
    errors: dict[str, str] | None = None,
    show_costs: bool = False,
    include_header: bool = True,
    total_only: bool = False,
) -> str:
    """Format token count results according to specified format.

    ``models`` is the order the models were asked for; the rows follow it rather than
    the order the counts happen to be stored in, and a model with no count keeps its
    place. ``errors`` maps such a model to why it has none, and reaches JSON alone.

    Raises:
        ValueError: If format is not supported
    """
    if output_format == OutputFormat.TEXT:
        return format_text(
            results, models=models, show_costs=show_costs, include_header=include_header
        )
    if output_format == OutputFormat.JSON:
        return format_json(results, models=models, errors=errors, total_only=total_only)
    if output_format == OutputFormat.CSV:
        return format_csv(
            results, models=models, include_header=include_header, show_costs=show_costs
        )
    if output_format == OutputFormat.TSV:
        return format_tsv(
            results, models=models, include_header=include_header, show_costs=show_costs
        )
    raise ValueError(f"Unknown format: {output_format}")


def _format_file_json(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    errors: dict[str, dict[str, str]],
    total_reasons: dict[str, str],
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
                "counts": _counts_payload(counts, models, errors.get(filename)),
            }
            for filename, counts in file_results.items()
        ]
    )
    return _envelope(results, _counts_payload(totals, models, total_reasons))


def _model_reasons(
    file_results: dict[str, dict[str, TokenCount]], errors: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Why each model has no total: the first failure it hit, in input order.

    Read in ``file_results`` order rather than in ``errors`` order, and called before
    the rows are sorted, so ``--sort`` cannot change which failure a total names.
    Fenced by test_the_total_names_the_first_failure_in_file_order_not_in_errors_order
    and by test_sorting_the_rows_does_not_change_which_failure_the_total_names, which
    give the two files failures that can be told apart.
    """
    reasons: dict[str, str] = {}
    for filename in file_results:
        for model, reason in errors.get(filename, {}).items():
            reasons.setdefault(model, reason)
    return reasons


def _collect_models(
    file_results: dict[str, dict[str, TokenCount]], requested: list[str] | None = None
) -> list[str]:
    # The one order a run's output uses: the table columns, the delimited columns and
    # both JSON arrays all read it, so nothing in a document has to be re-read against a
    # second order. `requested` is the order the models were asked for, and it is the
    # whole answer: every source lists every one of them. Without it the best available
    # order is first encounter, which is the same order right up until a model fails on
    # an early source and succeeds on a later one.
    if requested is None:
        return list(
            dict.fromkeys(model for counts in file_results.values() for model in counts)
        )
    # Every model asked for keeps its place, including one no source could be counted
    # for: it takes a column of empty cells rather than vanishing, because a user who
    # named a model is owed the news that it failed, and because a column set that
    # depends on what the counting produced cannot be parsed without reading it first.
    # Deduplicated, since `--model` is repeatable and two columns of one name cannot be
    # keyed apart: test_a_model_named_twice_takes_one_column.
    return list(dict.fromkeys(requested))


def _format_file_table_delimited(
    file_results: dict[str, dict[str, TokenCount]],
    *,
    models: list[str],
    separator: str,
    total_only: bool,
    include_header: bool,
    show_costs: bool,
) -> str:
    # Three columns per model, or two without --cost: the models named on the command
    # line and the flags beside them are the whole of what decides the header, so a
    # consumer knows the shape before the run and every run of one command matches.
    rows: list[list[str]] = []
    headers = ["file"]
    for model in models:
        headers.append(f"{model}_tokens")
        if show_costs:
            headers.append(f"{model}_cost")
        headers.append(f"{model}_approximate")

    if include_header:
        rows.append(headers)

    if total_only:
        totals = _compute_totals(file_results, models=models)
        total_row: list[str] = ["TOTAL"]
        for model in models:
            # .get, because a model no file could be counted for has no total; its
            # cells are empty, the same as a file that model missed.
            total_row.extend(_delimited_cells(totals.get(model), show_costs=show_costs))
        rows.append(total_row)
        return _render_delimited(rows, separator)

    for filename, model_counts in file_results.items():
        row: list[str] = [filename]
        for model in models:
            row.extend(_delimited_cells(model_counts.get(model), show_costs=show_costs))
        rows.append(row)

    return _render_delimited(rows, separator)


def _delimited_cells(counted: TokenCount | None, *, show_costs: bool) -> list[str]:
    if counted is None:
        # No count at all for this file and model. Every cell is empty rather than
        # holding a stand-in like N/A: a column a consumer reads as a number has to
        # hold numbers and blanks and nothing else. Why it is empty is in the JSON
        # `reason`, which is where free text can be quoted safely.
        cells = [""]
        if show_costs:
            cells.append("")
        cells.append("")
        return cells

    cells = [str(counted.count)]
    if show_costs:
        cells.append(format_cost_value(counted.cost))
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
            # .get: a model no file could be counted for has no total, and says so the
            # same way its rows do. This row and the ones above it are fenced by
            # test_the_file_table_says_na_for_a_model_no_file_could_be_counted_for and
            # test_the_file_table_says_na_in_the_cost_cells_it_has_no_count_for.
            total = totals.get(model)
            total_row.append("N/A" if total is None else f"{total.count:,}")
            if show_costs:
                total_row.append("N/A" if total is None else format_cost(total.cost))
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
    #
    # Normalized on the way out, because a running sum is a cost this module
    # produces: adding two exact per-file costs lands on 8.750000000000001e-06,
    # and every format has to report the one number.
    if addition is None:
        return running
    return normalize_cost(addition if running is None else running + addition)


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
    errors: dict[str, dict[str, str]] | None = None,
    show_costs: bool = False,
    include_header: bool = True,
    sort_order: SortOrder | str = SortOrder.INPUT,
) -> str:
    """Format per-file token counts with files as rows and models as columns.

    ``models`` is the order the models were asked for; the columns and the JSON arrays
    follow it rather than the order the counts happen to be stored in, and a model no
    file could be counted for keeps its column. Omit it and the order is first
    encounter in ``file_results``. ``errors`` maps a filename to why each model has no
    count for it, and reaches JSON alone.
    """
    models = _collect_models(file_results, models)
    errors = errors or {}
    total_reasons = _model_reasons(file_results, errors)
    if not total_only:
        # A total-only run prints no file rows to order, and sorting anyway would leak
        # into the one row it does print: _compute_totals joins the per-file caveats in
        # iteration order, so the row's caveat text would follow --sort.
        file_results = _sort_file_results(
            file_results, models=models, sort_order=sort_order
        )

    if output_format == OutputFormat.JSON:
        return _format_file_json(
            file_results,
            models=models,
            errors=errors,
            total_reasons=total_reasons,
            total_only=total_only,
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
