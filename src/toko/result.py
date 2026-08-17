"""The value a token count resolves to."""

from dataclasses import dataclass
from enum import StrEnum


class CaveatKind(StrEnum):
    """Why a count came from something other than the model's own tokenizer."""

    # An OpenAI name toko has no verified encoding for, counted with o200k_base.
    OPENAI_ENCODING_GUESS = "openai_encoding_guess"
    # An xAI model counted with the Grok-1 Hugging Face tokenizer, because the
    # xAI token API was unreachable.
    XAI_GROK1_STANDIN = "xai_grok1_standin"
    # A Mistral name mistral-common bundles no tokenizer for, counted with tekken.
    MISTRAL_TOKENIZER_FALLBACK = "mistral_tokenizer_fallback"


@dataclass(frozen=True, slots=True)
class Caveat:
    """One reason a count is not what the named model's own tokenizer would give.

    `message` is the sentence the CLI prints on stderr; the other fields carry the
    same facts in a form a program can branch on, so nothing has to be parsed out
    of the prose.
    """

    kind: CaveatKind
    model: str
    message: str
    # The tiktoken encoding an OPENAI_ENCODING_GUESS was made with.
    encoding: str | None = None
    # The stand-in tokenizer that produced the count, where one was substituted.
    tokenizer: str | None = None
    # What made the exact path unavailable, for the caveats that have a cause to
    # report (the xAI API error, redacted).
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Retirement:
    """A model the provider has retired, and what it serves in its place."""

    model: str
    # ISO date the provider retired it, or None when no date was ever published.
    date: str | None = None
    # Set when the provider still answers for the name but serves another model,
    # which makes the count that model's rather than the named one's.
    redirects_to: str | None = None


@dataclass(frozen=True, slots=True)
class TokenCount:
    count: int
    # The canonical name the request resolved to, which need not be the string the
    # caller passed: aliases, casing, and Google's "models/" prefix all resolve here.
    model: str
    provider: str
    approximate: bool = False
    caveats: tuple[Caveat, ...] = ()
    cost: float | None = None
    retirement: Retirement | None = None
