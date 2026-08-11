"""Tests for output formatters."""

import csv
import io
import json

from toko.formatters import format_file_table, format_output
from toko.result import TokenCount


def _counted(count: int, *, cost: float | None = None, model: str = "gpt-5"):
    return TokenCount(count=count, model=model, provider="openai", cost=cost)


def test_json_matches_plain_mapping_without_costs():
    payload = json.loads(format_output({"gpt-5": _counted(12)}, output_format="json"))
    assert payload == {"gpt-5": 12}


def test_json_includes_costs_when_requested():
    payload = json.loads(
        format_output(
            {"gpt-5": _counted(12, cost=0.000015)},
            output_format="json",
            show_costs=True,
        )
    )
    assert payload == {"gpt-5": {"tokens": 12, "cost": 0.000015}}


def test_json_reports_unpriced_model_as_null_cost():
    payload = json.loads(
        format_output(
            {"mystery-model": _counted(3, model="mystery-model")},
            output_format="json",
            show_costs=True,
        )
    )
    assert payload == {"mystery-model": {"tokens": 3, "cost": None}}


def test_json_costs_cover_every_model():
    results = {
        "gpt-5": _counted(12, cost=0.0001),
        "gpt-4.1": _counted(14, cost=0.0002, model="gpt-4.1"),
    }
    payload = json.loads(format_output(results, output_format="json", show_costs=True))
    assert payload == {
        "gpt-5": {"tokens": 12, "cost": 0.0001},
        "gpt-4.1": {"tokens": 14, "cost": 0.0002},
    }


def test_json_reports_a_caveat_on_an_exact_count():
    """A caveat must not need an approximate sibling to become visible."""
    counted = TokenCount(
        count=7, model="gpt-5", provider="openai", caveat="retired last spring"
    )
    payload = json.loads(format_output({"gpt-5": counted}, output_format="json"))
    assert payload == {"gpt-5": {"tokens": 7, "caveat": "retired last spring"}}


def test_file_json_keeps_one_shape_when_only_one_file_has_a_caveat():
    file_results = {
        "a.txt": {
            "gpt-5": TokenCount(
                count=7, model="gpt-5", provider="openai", caveat="retired last spring"
            )
        },
        "b.txt": {"gpt-5": _counted(4)},
    }
    payload = json.loads(format_file_table(file_results, output_format="json"))
    assert payload == {
        "a.txt": {"gpt-5": {"tokens": 7, "caveat": "retired last spring"}},
        "b.txt": {"gpt-5": {"tokens": 4}},
    }


def _caveated_column(count: int, caveat: str):
    return {
        "grok-4.5": TokenCount(
            count=count,
            model="grok-4.5",
            provider="xai",
            approximate=True,
            caveat=caveat,
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
    assert payload == {
        "grok-4.5": {
            "tokens": 7,
            "approximate": True,
            "caveat": (
                "the xAI token API was unavailable (timeout); "
                "the xAI token API was unavailable (503)"
            ),
        }
    }


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
    assert payload == {"grok-4.5": {"tokens": 7, "approximate": True, "caveat": caveat}}


def test_file_json_matches_plain_mapping_without_costs():
    payload = json.loads(
        format_file_table({"a.txt": {"gpt-5": _counted(4)}}, output_format="json")
    )
    assert payload == {"a.txt": {"gpt-5": 4}}


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
    assert payload == {
        "a.txt": {"gpt-5": {"tokens": 4, "cost": 0.0002}},
        "b.txt": {"gpt-5": {"tokens": 9, "cost": None}},
    }


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
    assert payload == {"mystery-model": {"tokens": 13, "cost": None}}


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
    assert payload == {"gpt-5": {"tokens": 13, "cost": 0.0002}}


def test_total_only_csv_marks_a_wholly_unpriced_model_not_available():
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
    assert output == "TOTAL,4,$0.0002,5,N/A"


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
    assert output == ("file,gpt-5_tokens,gpt-5_cost\na.txt,4,$0.0002\nb.txt,9,N/A")


def test_tsv_leaves_a_comma_in_the_path_alone():
    output = format_file_table({"a,b.txt": {"gpt-5": _counted(4)}}, output_format="tsv")
    assert output == "file\tgpt-5\na,b.txt\t4"


def test_total_only_tsv_matches_the_per_file_marker_for_a_missing_cost():
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
    assert per_file == ["a.txt\t4\tN/A", "b.txt\t9\tN/A"]
    assert total == "TOTAL\t13\tN/A"
