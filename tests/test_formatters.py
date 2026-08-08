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
