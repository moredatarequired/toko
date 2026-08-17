"""Tests for output formatters."""

import csv
import io
import json
import os
import re

import pytest

from tests.pty_runner import HAS_PTY, PTY_SKIP_REASON, run_under_pty
from toko.formatters import SCHEMA_VERSION, format_file_table, format_output
from toko.result import Caveat, CaveatKind, Retirement, TokenCount


def _counted(count: int, *, cost: float | None = None, model: str = "gpt-5"):
    return TokenCount(count=count, model=model, provider="openai", cost=cost)


def _expected(model: str, tokens: int, **overrides):
    """Build the count object every JSON document carries, keys always present."""
    count = {
        "model": model,
        "tokens": tokens,
        "approximate": False,
        "cost": None,
        "caveats": [],
        "retirement": None,
    }
    return count | overrides


def _text_envelope(*counts):
    return {
        "schema_version": SCHEMA_VERSION,
        "results": [{"source": {"kind": "text", "name": None}, "counts": list(counts)}],
        "totals": list(counts),
    }


def test_json_wraps_a_text_run_in_the_envelope():
    payload = json.loads(format_output({"gpt-5": _counted(12)}, output_format="json"))
    assert payload == _text_envelope(_expected("gpt-5", 12))


def test_json_includes_costs_when_requested():
    payload = json.loads(
        format_output(
            {"gpt-5": _counted(12, cost=0.000015)},
            output_format="json",
            show_costs=True,
        )
    )
    assert payload == _text_envelope(_expected("gpt-5", 12, cost=0.000015))


def test_json_reports_unpriced_model_as_null_cost():
    payload = json.loads(
        format_output(
            {"mystery-model": _counted(3, model="mystery-model")},
            output_format="json",
            show_costs=True,
        )
    )
    assert payload == _text_envelope(_expected("mystery-model", 3))


def test_json_costs_cover_every_model():
    results = {
        "gpt-5": _counted(12, cost=0.0001),
        "gpt-4.1": _counted(14, cost=0.0002, model="gpt-4.1"),
    }
    payload = json.loads(format_output(results, output_format="json", show_costs=True))
    assert payload == _text_envelope(
        _expected("gpt-5", 12, cost=0.0001), _expected("gpt-4.1", 14, cost=0.0002)
    )


def _source_names(payload) -> list[str]:
    return [result["source"]["name"] for result in payload["results"]]


def _caveat(message: str, *, model: str = "grok-4.5") -> Caveat:
    return Caveat(
        kind=CaveatKind.XAI_GROK1_STANDIN,
        model=model,
        message=message,
        tokenizer="Xenova/grok-1-tokenizer",
        reason=message,
    )


def _expected_caveat(message: str, *, model: str = "grok-4.5"):
    return {
        "kind": "xai_grok1_standin",
        "model": model,
        "message": message,
        "encoding": None,
        "tokenizer": "Xenova/grok-1-tokenizer",
        "reason": message,
    }


def test_json_reports_a_caveat_on_an_exact_count():
    """A caveat must not need an approximate sibling to become visible."""
    counted = TokenCount(
        count=7,
        model="gpt-5",
        provider="openai",
        caveats=(_caveat("stood in for it", model="gpt-5"),),
    )
    payload = json.loads(format_output({"gpt-5": counted}, output_format="json"))
    assert payload == _text_envelope(
        _expected(
            "gpt-5", 7, caveats=[_expected_caveat("stood in for it", model="gpt-5")]
        )
    )


def test_json_reports_a_retired_model_as_an_object():
    counted = TokenCount(
        count=7,
        model="grok-3",
        provider="xai",
        retirement=Retirement(
            model="grok-3", date="2026-06-10", redirects_to="grok-4.3"
        ),
    )
    payload = json.loads(format_output({"grok-3": counted}, output_format="json"))
    assert payload["totals"][0]["retirement"] == {
        "model": "grok-3",
        "date": "2026-06-10",
        "redirects_to": "grok-4.3",
    }


def test_file_json_gives_every_count_the_same_keys():
    file_results = {
        "a.txt": {
            "gpt-5": TokenCount(
                count=7,
                model="gpt-5",
                provider="openai",
                caveats=(_caveat("stood in for it", model="gpt-5"),),
            )
        },
        "b.txt": {"gpt-5": _counted(4)},
    }
    payload = json.loads(format_file_table(file_results, output_format="json"))
    assert [result["source"] for result in payload["results"]] == [
        {"kind": "file", "name": "a.txt"},
        {"kind": "file", "name": "b.txt"},
    ]
    assert [
        set(count) for result in payload["results"] for count in result["counts"]
    ] == [set(_expected("gpt-5", 0)), set(_expected("gpt-5", 0))]


def test_file_json_distinguishes_a_url_from_a_path():
    payload = json.loads(
        format_file_table(
            {
                "https://example.com/a.txt": {"gpt-5": _counted(4)},
                "a.txt": {"gpt-5": _counted(4)},
            },
            output_format="json",
        )
    )
    assert [result["source"] for result in payload["results"]] == [
        {"kind": "url", "name": "https://example.com/a.txt"},
        {"kind": "file", "name": "a.txt"},
    ]


def _caveated_column(count: int, caveat: str):
    return {
        "grok-4.5": TokenCount(
            count=count,
            model="grok-4.5",
            provider="xai",
            approximate=True,
            caveats=(_caveat(caveat),),
        )
    }


def test_total_only_json_keeps_every_distinct_caveat():
    """Files can fail differently, and the total must not hide the later failures."""
    payload = json.loads(
        format_file_table(
            {
                "a.txt": _caveated_column(
                    3, "the xAI token API was unavailable (timeout)"
                ),
                "b.txt": _caveated_column(4, "the xAI token API was unavailable (503)"),
            },
            output_format="json",
            total_only=True,
        )
    )
    assert payload["results"] == []
    assert payload["totals"] == [
        _expected(
            "grok-4.5",
            7,
            approximate=True,
            caveats=[
                _expected_caveat("the xAI token API was unavailable (timeout)"),
                _expected_caveat("the xAI token API was unavailable (503)"),
            ],
        )
    ]


def test_total_only_json_reports_a_column_wide_caveat_once():
    caveat = "counted with the Grok-1 tokenizer"
    payload = json.loads(
        format_file_table(
            {
                "a.txt": _caveated_column(3, caveat),
                "b.txt": _caveated_column(4, caveat),
            },
            output_format="json",
            total_only=True,
        )
    )
    assert payload["totals"] == [
        _expected("grok-4.5", 7, approximate=True, caveats=[_expected_caveat(caveat)])
    ]


def test_file_json_wraps_one_file_in_the_envelope():
    payload = json.loads(
        format_file_table({"a.txt": {"gpt-5": _counted(4)}}, output_format="json")
    )
    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "results": [
            {
                "source": {"kind": "file", "name": "a.txt"},
                "counts": [_expected("gpt-5", 4)],
            }
        ],
        "totals": [_expected("gpt-5", 4)],
    }


def test_file_json_includes_costs_when_requested():
    payload = json.loads(
        format_file_table(
            {
                "a.txt": {"gpt-5": _counted(4, cost=0.0002)},
                "b.txt": {"gpt-5": _counted(9)},
            },
            output_format="json",
            show_costs=True,
        )
    )
    assert payload["results"] == [
        {
            "source": {"kind": "file", "name": "a.txt"},
            "counts": [_expected("gpt-5", 4, cost=0.0002)],
        },
        {
            "source": {"kind": "file", "name": "b.txt"},
            "counts": [_expected("gpt-5", 9)],
        },
    ]
    assert payload["totals"] == [_expected("gpt-5", 13, cost=0.0002)]


def test_total_only_json_reports_wholly_unpriced_model_as_null():
    payload = json.loads(
        format_file_table(
            {
                "a.txt": {"mystery-model": _counted(4, model="mystery-model")},
                "b.txt": {"mystery-model": _counted(9, model="mystery-model")},
            },
            output_format="json",
            total_only=True,
            show_costs=True,
        )
    )
    assert payload["totals"] == [_expected("mystery-model", 13)]


def test_total_only_json_sums_only_the_files_it_could_price():
    payload = json.loads(
        format_file_table(
            {
                "a.txt": {"gpt-5": _counted(4, cost=0.0002)},
                "b.txt": {"gpt-5": _counted(9)},
            },
            output_format="json",
            total_only=True,
            show_costs=True,
        )
    )
    assert payload["totals"] == [_expected("gpt-5", 13, cost=0.0002)]


def test_total_only_csv_leaves_a_wholly_unpriced_cost_empty():
    output = format_file_table(
        {
            "a.txt": {
                "gpt-5": _counted(4, cost=0.0002),
                "mystery-model": _counted(5, model="mystery-model"),
            }
        },
        output_format="csv",
        total_only=True,
        show_costs=True,
        include_header=False,
    )
    assert output == "TOTAL,4,0.0002,5,"


def _csv_rows(output: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(output)))


def test_csv_quotes_a_comma_in_a_model_name():
    output = format_output(
        {"gpt-5, preview": _counted(12, model="gpt-5, preview")}, output_format="csv"
    )
    assert output == 'model,tokens\n"gpt-5, preview",12'
    assert _csv_rows(output)[1] == ["gpt-5, preview", "12"]


def test_csv_file_rows_survive_a_comma_in_the_path():
    output = format_file_table({"a,b.txt": {"gpt-5": _counted(4)}}, output_format="csv")
    assert _csv_rows(output) == [["file", "gpt-5"], ["a,b.txt", "4"]]


def test_csv_file_rows_survive_a_quote_in_the_path():
    output = format_file_table(
        {'we"ird.txt': {"gpt-5": _counted(4)}}, output_format="csv"
    )
    assert _csv_rows(output) == [["file", "gpt-5"], ['we"ird.txt', "4"]]


def test_csv_file_rows_survive_a_newline_in_the_path():
    output = format_file_table(
        {"two\nlines.txt": {"gpt-5": _counted(4)}}, output_format="csv"
    )
    assert _csv_rows(output) == [["file", "gpt-5"], ["two\nlines.txt", "4"]]


def test_csv_keeps_ordinary_rows_unquoted():
    output = format_file_table(
        {"a.txt": {"gpt-5": _counted(4, cost=0.0002)}, "b.txt": {"gpt-5": _counted(9)}},
        output_format="csv",
        show_costs=True,
    )
    assert output == ("file,gpt-5_tokens,gpt-5_cost\na.txt,4,0.0002\nb.txt,9,")


def test_tsv_leaves_a_comma_in_the_path_alone():
    output = format_file_table({"a,b.txt": {"gpt-5": _counted(4)}}, output_format="tsv")
    assert output == "file\tgpt-5\na,b.txt\t4"


def test_total_only_tsv_matches_the_per_file_cell_for_a_missing_cost():
    file_results = {
        "a.txt": {"mystery-model": _counted(4, model="mystery-model")},
        "b.txt": {"mystery-model": _counted(9, model="mystery-model")},
    }
    per_file = format_file_table(
        file_results, output_format="tsv", show_costs=True, include_header=False
    ).splitlines()
    total = format_file_table(
        file_results,
        output_format="tsv",
        total_only=True,
        show_costs=True,
        include_header=False,
    )
    assert per_file == ["a.txt\t4\t", "b.txt\t9\t"]
    assert total == "TOTAL\t13\t"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The formatters force a terminal so rich still styles the cells; the layout is what
# these assert, so the colors come off first.
_BOX_DRAWING = re.compile(r"[─-╿]")


def _plain_lines(output: str) -> list[str]:
    return _ANSI.sub("", output).splitlines()


def test_text_table_has_no_borders():
    output = format_output(
        {"gpt-5": _counted(12), "gpt-4.1": _counted(1234, model="gpt-4.1")}
    )
    assert _plain_lines(output) == [
        "Model    Tokens",
        "gpt-5        12",
        "gpt-4.1   1,234",
    ]


def test_text_table_without_a_header_starts_at_the_first_row():
    output = format_output(
        {"gpt-5": _counted(12), "gpt-4.1": _counted(1234, model="gpt-4.1")},
        include_header=False,
    )
    assert _plain_lines(output) == ["gpt-5       12", "gpt-4.1  1,234"]


def test_text_table_header_is_bold(monkeypatch):
    # Without a rule under it, bold is all that separates the header from the data, so
    # this test keeps the ANSI. rich drops every style under TERM=dumb/unknown, hence
    # the pinned terminal.
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    header, *rows = format_output(
        {"gpt-5": _counted(12), "gpt-4.1": _counted(1234, model="gpt-4.1")}
    ).splitlines()
    assert _ANSI.sub("", header) == "Model    Tokens"
    assert header.startswith("\x1b[1m")
    assert "\x1b[1m" not in "".join(rows)


def test_file_text_table_has_no_borders():
    output = format_file_table(
        {"a.txt": {"gpt-5": _counted(4)}, "longer/b.txt": {"gpt-5": _counted(1234)}}
    )
    assert _plain_lines(output) == [
        "File          gpt-5",
        "a.txt             4",
        "longer/b.txt  1,234",
        "TOTAL         1,238",
    ]


def test_file_text_table_without_a_header_starts_at_the_first_row():
    output = format_file_table(
        {"a.txt": {"gpt-5": _counted(4)}, "longer/b.txt": {"gpt-5": _counted(1234)}},
        include_header=False,
    )
    assert _plain_lines(output) == [
        "a.txt             4",
        "longer/b.txt  1,234",
        "TOTAL         1,238",
    ]


@pytest.mark.parametrize("show_costs", [False, True])
@pytest.mark.parametrize("total_only", [False, True])
@pytest.mark.parametrize("include_header", [False, True])
def test_every_text_table_variant_is_unruled_and_flush_left(
    show_costs, total_only, include_header
):
    output = format_file_table(
        {
            "a.txt": {"gpt-5": _counted(4, cost=0.0002)},
            "b.txt": {"gpt-5": _counted(9), "gpt-4.1": _counted(11, model="gpt-4.1")},
        },
        total_only=total_only,
        show_costs=show_costs,
        include_header=include_header,
    )
    lines = _plain_lines(output)
    assert not _BOX_DRAWING.search("\n".join(lines))
    # With costs the header spans two lines and its first one is blank under "File", so
    # the row that proves nothing indents the body is the last one.
    assert lines[-1].startswith("TOTAL")


def _row_labels(output: str) -> list[str]:
    # The first cell of every row the table drew, with the borders taken out so the
    # assertion is about the order of the rows and not the shape of the frame.
    words = [_BOX_DRAWING.sub(" ", line).split() for line in _plain_lines(output)]
    return [row[0] for row in words if row]


def _three_files() -> dict[str, dict[str, TokenCount]]:
    return {
        "a.txt": {"gpt-5": _counted(1)},
        "b.txt": {"gpt-5": _counted(30)},
        "c.txt": {"gpt-5": _counted(7)},
    }


def _mixed_paths() -> dict[str, dict[str, TokenCount]]:
    # Insertion order is neither path order nor count order, so each --sort value lands
    # on a different arrangement and no test can pass by accident.
    return {
        "src/b.txt": {"gpt-5": _counted(30)},
        "a.txt": {"gpt-5": _counted(7)},
        "src/a.txt": {"gpt-5": _counted(1)},
    }


def test_rows_keep_their_input_order_by_default():
    output = format_file_table(_three_files(), output_format="csv")
    assert _csv_rows(output)[1:] == [["a.txt", "1"], ["b.txt", "30"], ["c.txt", "7"]]


def test_input_sort_is_the_default():
    assert format_file_table(
        _mixed_paths(), output_format="csv", sort_order="input"
    ) == format_file_table(_mixed_paths(), output_format="csv")


def test_input_sort_leaves_paths_that_are_out_of_order_alone():
    output = format_file_table(_mixed_paths(), output_format="csv", sort_order="input")
    assert [row[0] for row in _csv_rows(output)[1:]] == [
        "src/b.txt",
        "a.txt",
        "src/a.txt",
    ]


def test_path_sort_orders_rows_by_the_path_string():
    # A plain string sort, which is what keeps src/a.txt next to src/b.txt.
    output = format_file_table(_mixed_paths(), output_format="csv", sort_order="path")
    assert [row[0] for row in _csv_rows(output)[1:]] == [
        "a.txt",
        "src/a.txt",
        "src/b.txt",
    ]


def test_path_sort_reorders_json_results_the_same_way():
    payload = json.loads(
        format_file_table(_mixed_paths(), output_format="json", sort_order="path")
    )
    assert _source_names(payload) == ["a.txt", "src/a.txt", "src/b.txt"]


def test_path_sort_reorders_tsv_the_same_way():
    output = format_file_table(
        _mixed_paths(), output_format="tsv", sort_order="path", include_header=False
    )
    assert output.splitlines() == ["a.txt\t7", "src/a.txt\t1", "src/b.txt\t30"]


def test_path_sort_reorders_the_text_table_and_leaves_the_total_last():
    output = format_file_table(_mixed_paths(), sort_order="path", include_header=False)
    assert _row_labels(output) == ["a.txt", "src/a.txt", "src/b.txt", "TOTAL"]


def test_count_sort_puts_the_largest_file_first():
    output = format_file_table(_three_files(), output_format="csv", sort_order="count")
    assert _csv_rows(output)[1:] == [["b.txt", "30"], ["c.txt", "7"], ["a.txt", "1"]]


def test_count_sort_reorders_json_results_the_same_way():
    payload = json.loads(
        format_file_table(_three_files(), output_format="json", sort_order="count")
    )
    assert _source_names(payload) == ["b.txt", "c.txt", "a.txt"]


def test_count_sort_reorders_the_text_table_and_leaves_the_total_last():
    output = format_file_table(_three_files(), sort_order="count", include_header=False)
    assert _row_labels(output) == ["b.txt", "c.txt", "a.txt", "TOTAL"]


def test_count_sort_ranks_by_the_leftmost_model_column():
    # Columns are ordered by model name, so the leftmost one here is claude-opus-4-5,
    # and ranking by gpt-5 instead would put b.txt first.
    file_results = {
        "a.txt": {
            "gpt-5": _counted(1),
            "claude-opus-4-5": _counted(50, model="claude-opus-4-5"),
        },
        "b.txt": {
            "gpt-5": _counted(90),
            "claude-opus-4-5": _counted(2, model="claude-opus-4-5"),
        },
    }
    output = format_file_table(file_results, output_format="csv", sort_order="count")
    assert _csv_rows(output) == [
        ["file", "claude-opus-4-5", "gpt-5"],
        ["a.txt", "50", "1"],
        ["b.txt", "2", "90"],
    ]


def test_count_sort_puts_a_file_the_leading_model_missed_last():
    file_results = {
        "missed.txt": {"gpt-5": _counted(3)},
        "counted.txt": {
            "gpt-5": _counted(4),
            "claude-opus-4-5": _counted(1, model="claude-opus-4-5"),
        },
    }
    output = format_file_table(file_results, output_format="csv", sort_order="count")
    assert _csv_rows(output)[1:] == [
        ["counted.txt", "1", "4"],
        ["missed.txt", "N/A", "3"],
    ]


def test_count_sort_breaks_a_tie_on_the_path():
    file_results = {
        "z.txt": {"gpt-5": _counted(5)},
        "m.txt": {"gpt-5": _counted(5)},
        "a.txt": {"gpt-5": _counted(5)},
    }
    output = format_file_table(file_results, output_format="csv", sort_order="count")
    assert [row[0] for row in _csv_rows(output)[1:]] == ["a.txt", "m.txt", "z.txt"]


def _differently_caveated_files() -> dict[str, dict[str, TokenCount]]:
    # Two files that failed differently, the larger count second so a count sort would
    # swap them and a path sort would not.
    return {
        "b.txt": _caveated_column(3, "the xAI token API was unavailable (429)"),
        "a.txt": _caveated_column(400, "the xAI token API was unavailable (503)"),
    }


def test_sort_leaves_a_total_only_run_alone():
    """--sort orders the per-file rows, so it must not rewrite the TOTAL row's caveat.

    _compute_totals joins the per-file caveats in iteration order, so sorting a run that
    prints no file rows would still show through, in the one row it does print.
    """
    unsorted = format_file_table(
        _differently_caveated_files(), output_format="json", total_only=True
    )
    for sort_order in ("input", "path", "count"):
        assert (
            format_file_table(
                _differently_caveated_files(),
                output_format="json",
                total_only=True,
                sort_order=sort_order,
            )
            == unsorted
        )
    # Not vacuous: the caveats are in the order the files arrived, which is the order
    # both other values would have changed.
    assert [
        caveat["message"] for caveat in json.loads(unsorted)["totals"][0]["caveats"]
    ] == [
        "the xAI token API was unavailable (429)",
        "the xAI token API was unavailable (503)",
    ]


def test_an_unknown_sort_order_is_rejected():
    with pytest.raises(ValueError, match="Unknown sort order: size"):
        format_file_table(_three_files(), output_format="csv", sort_order="size")


# TERM must not change the layout: rich reports a hardcoded 80x25 for a dumb terminal
# unless the console is given both a width and a height (rich.console.Console.size).
# These tests drop LINES because rich takes it as that height in Console.__init__, which
# satisfies the both-dimensions case and hides the bug entirely.


def test_file_text_table_keeps_its_width_under_a_dumb_terminal(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("LINES", raising=False)
    wide_name = f"{'nested/' * 15}a_file_with_a_long_basename.txt"
    assert len(wide_name) > 80

    output = format_file_table({wide_name: {"gpt-5": _counted(4)}})

    assert wide_name in _ANSI.sub("", output)


def test_model_table_width_follows_the_terminal_not_term(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("LINES", raising=False)
    # A console that reads FORCE_COLOR is a terminal whatever it writes to, so this keeps
    # the size probe honest: it has to opt out of being a terminal to escape TERM=dumb.
    monkeypatch.setenv("FORCE_COLOR", "1")
    wide_model = f"a-model-with-an-implausibly-long-name{'-and-more' * 10}"
    assert len(wide_model) > 80
    results = {wide_model: _counted(12, model=wide_model), "gpt-5": _counted(1234)}

    monkeypatch.setenv("TERM", "dumb")
    dumb = _plain_lines(format_output(results))
    monkeypatch.setenv("TERM", "xterm-256color")
    smart = _plain_lines(format_output(results))

    assert dumb == smart
    assert any(wide_model in line for line in dumb)


def test_a_dumb_terminal_still_gets_no_escape_sequences(monkeypatch):
    """Sizing the table ourselves must not cost rich its own reading of TERM."""
    monkeypatch.delenv("LINES", raising=False)
    results = {"gpt-5": _counted(1234), "gpt-5-mini": _counted(12, model="gpt-5-mini")}

    monkeypatch.setenv("TERM", "dumb")
    dumb = format_output(results)
    monkeypatch.setenv("TERM", "xterm-256color")
    smart = format_output(results)

    assert "\x1b[" not in dumb
    # Otherwise the assertion above would hold for a table that is never styled at all.
    assert "\x1b[" in smart


@pytest.mark.skipif(not HAS_PTY, reason=PTY_SKIP_REASON)
def test_the_cli_file_table_fills_a_wide_dumb_terminal(tmp_path):
    """End to end: TERM=dumb must not shrink the file table off its pinned 200 columns.

    The terminal is only there to keep the CLI on the text format; its own width never
    reaches this table, so the assertions below hold at any window size.
    """
    nested = tmp_path / "a_directory_whose_name_pads_the_paths_past_eighty_columns"
    nested.mkdir()
    first = nested / "first_file_with_a_long_basename.txt"
    second = nested / "second_file_with_a_long_basename.txt"
    first.write_text("hello world one\n")
    second.write_text("hello world two three four\n")
    # Otherwise an 80-column fallback would render these in full and prove nothing.
    assert len(str(nested)) > 80

    script = tmp_path / "file_table_driver.py"
    script.write_text(
        f"from toko.cli import app\n\napp([{str(first)!r}, {str(second)!r}])\n"
    )
    env = dict(os.environ, TERM="dumb")
    env.pop("LINES", None)
    env.pop("COLUMNS", None)

    output = _ANSI.sub("", run_under_pty(str(script), env))

    assert first.name in output
    assert second.name in output
    rows = [line for line in output.splitlines() if ".txt" in line]
    assert len(rows) == 2
    assert rows[0] != rows[1]


@pytest.mark.skipif(not HAS_PTY, reason=PTY_SKIP_REASON)
def test_the_model_table_uses_the_terminal_on_stdin_when_stdout_is_a_pipe(tmp_path):
    """A redirected stdout must not cost us the terminal that stdin and stderr still are."""
    wide_model = f"a-model-with-an-implausibly-long-name{'-and-more' * 10}"
    assert len(wide_model) > 80

    script = tmp_path / "model_table_driver.py"
    script.write_text(
        "from toko.formatters import format_output\n"
        "from toko.result import TokenCount\n\n"
        f"model = {wide_model!r}\n"
        # A lone result prints as a bare count, so there has to be a second row.
        "print(\n"
        "    format_output(\n"
        "        {\n"
        "            model: TokenCount(count=12, model=model, provider='openai'),\n"
        "            'gpt-5': TokenCount(count=1234, model='gpt-5', provider='openai'),\n"
        "        }\n"
        "    )\n"
        ")\n"
    )
    env = dict(os.environ, TERM="xterm-256color")
    env.pop("LINES", None)
    env.pop("COLUMNS", None)

    output = _ANSI.sub("", run_under_pty(str(script), env, pipe_stdout=True))

    assert wide_model in output
    assert "…" not in output
    assert max(len(line) for line in output.splitlines()) > 80
