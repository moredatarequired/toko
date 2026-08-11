"""Tests for output formatters."""

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
