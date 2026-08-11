"""Output format enum, kept in a leaf module so importing it stays cheap."""

from enum import StrEnum


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
