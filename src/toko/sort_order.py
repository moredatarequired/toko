"""Row ordering for the per-file table, kept in a leaf module so importing it stays cheap."""

from enum import StrEnum


class SortOrder(StrEnum):
    INPUT = "input"
    PATH = "path"
    COUNT = "count"
