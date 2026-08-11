"""Tests for output formatters."""

import json

from toko.formatters import format_file_table, format_output


def test_json_matches_plain_mapping_without_costs():
    payload = json.loads(format_output({"gpt-5": 12}, output_format="json"))
    assert payload == {"gpt-5": 12}


def test_json_includes_costs_when_requested():
    payload = json.loads(
        format_output({"gpt-5": 12}, output_format="json", costs={"gpt-5": 0.000015})
    )
    assert payload == {"gpt-5": {"tokens": 12, "cost": 0.000015}}


def test_json_reports_unpriced_model_as_null_cost():
    payload = json.loads(
        format_output(
            {"mystery-model": 3}, output_format="json", costs={"mystery-model": None}
        )
    )
    assert payload == {"mystery-model": {"tokens": 3, "cost": None}}


def test_json_costs_cover_every_model():
    results = {"gpt-5": 12, "gpt-4.1": 14}
    costs: dict[str, float | None] = {"gpt-5": 0.0001, "gpt-4.1": 0.0002}
    payload = json.loads(format_output(results, output_format="json", costs=costs))
    assert payload == {
        "gpt-5": {"tokens": 12, "cost": 0.0001},
        "gpt-4.1": {"tokens": 14, "cost": 0.0002},
    }


def test_file_json_matches_plain_mapping_without_costs():
    payload = json.loads(
        format_file_table({"a.txt": {"gpt-5": 4}}, output_format="json")
    )
    assert payload == {"a.txt": {"gpt-5": 4}}


def test_file_json_includes_costs_when_requested():
    payload = json.loads(
        format_file_table(
            {"a.txt": {"gpt-5": 4}, "b.txt": {"gpt-5": 9}},
            output_format="json",
            costs={"a.txt": {"gpt-5": 0.0002}, "b.txt": {"gpt-5": None}},
        )
    )
    assert payload == {
        "a.txt": {"gpt-5": {"tokens": 4, "cost": 0.0002}},
        "b.txt": {"gpt-5": {"tokens": 9, "cost": None}},
    }


def test_total_only_json_reports_wholly_unpriced_model_as_null():
    payload = json.loads(
        format_file_table(
            {"a.txt": {"mystery-model": 4}, "b.txt": {"mystery-model": 9}},
            output_format="json",
            total_only=True,
            costs={"a.txt": {"mystery-model": None}, "b.txt": {"mystery-model": None}},
        )
    )
    assert payload == {"mystery-model": {"tokens": 13, "cost": None}}


def test_total_only_json_sums_only_the_files_it_could_price():
    payload = json.loads(
        format_file_table(
            {"a.txt": {"gpt-5": 4}, "b.txt": {"gpt-5": 9}},
            output_format="json",
            total_only=True,
            costs={"a.txt": {"gpt-5": 0.0002}, "b.txt": {"gpt-5": None}},
        )
    )
    assert payload == {"gpt-5": {"tokens": 13, "cost": 0.0002}}


def test_total_only_csv_marks_a_wholly_unpriced_model_not_available():
    output = format_file_table(
        {"a.txt": {"gpt-5": 4, "mystery-model": 5}},
        output_format="csv",
        total_only=True,
        costs={"a.txt": {"gpt-5": 0.0002, "mystery-model": None}},
        include_header=False,
    )
    assert output == "TOTAL,4,$0.0002,5,N/A"


def test_total_only_tsv_matches_the_per_file_marker_for_a_missing_cost():
    file_results = {"a.txt": {"mystery-model": 4}, "b.txt": {"mystery-model": 9}}
    costs: dict[str, dict[str, float | None]] = {
        "a.txt": {"mystery-model": None},
        "b.txt": {"mystery-model": None},
    }
    per_file = format_file_table(
        file_results, output_format="tsv", costs=costs, include_header=False
    ).splitlines()
    total = format_file_table(
        file_results,
        output_format="tsv",
        total_only=True,
        costs=costs,
        include_header=False,
    )
    assert per_file == ["a.txt\t4\tN/A", "b.txt\t9\tN/A"]
    assert total == "TOTAL\t13\tN/A"
