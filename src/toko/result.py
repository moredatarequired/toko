"""The value a token count resolves to."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenCount:
    count: int
    # The canonical name the request resolved to, which need not be the string the
    # caller passed: aliases, casing, and Google's "models/" prefix all resolve here.
    model: str
    provider: str
    approximate: bool = False
    caveat: str | None = None
    cost: float | None = None
