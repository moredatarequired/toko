"""Tests for output formatters."""

import csv
import io
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


def _csv_rows(output: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(output)))


def test_csv_quotes_a_comma_in_a_model_name():
    output = format_output({"gpt-5, preview": 12}, output_format="csv")
    assert output == 'model,tokens\n"gpt-5, preview",12'
    assert _csv_rows(output)[1] == ["gpt-5, preview", "12"]


def test_csv_file_rows_survive_a_comma_in_the_path():
    output = format_file_table({"a,b.txt": {"gpt-5": 4}}, output_format="csv")
    assert _csv_rows(output) == [["file", "gpt-5"], ["a,b.txt", "4"]]


def test_csv_file_rows_survive_a_quote_in_the_path():
    output = format_file_table({'we"ird.txt': {"gpt-5": 4}}, output_format="csv")
    assert _csv_rows(output) == [["file", "gpt-5"], ['we"ird.txt', "4"]]


def test_csv_file_rows_survive_a_newline_in_the_path():
    output = format_file_table({"two\nlines.txt": {"gpt-5": 4}}, output_format="csv")
    assert _csv_rows(output) == [["file", "gpt-5"], ["two\nlines.txt", "4"]]


def test_csv_keeps_ordinary_rows_unquoted():
    output = format_file_table(
        {"a.txt": {"gpt-5": 4}, "b.txt": {"gpt-5": 9}},
        output_format="csv",
        costs={"a.txt": {"gpt-5": 0.0002}, "b.txt": {"gpt-5": None}},
    )
    assert output == ("file,gpt-5_tokens,gpt-5_cost\na.txt,4,$0.0002\nb.txt,9,N/A")


def test_tsv_leaves_a_comma_in_the_path_alone():
    output = format_file_table({"a,b.txt": {"gpt-5": 4}}, output_format="tsv")
    assert output == "file\tgpt-5\na,b.txt\t4"


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
