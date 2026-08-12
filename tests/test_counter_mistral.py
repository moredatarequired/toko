"""Tests for Mistral token counting against the installed mistral-common."""

import re
import warnings
from pathlib import Path

import pytest

from toko.counter import (
    MISTRAL_FALLBACK_TOKENIZER,
    MISTRAL_TOKENIZERS,
    _load_mistral_tokenizer,
    count_tokens,
)
from toko.models import OPTIONAL_GROUPS

tokenizers = pytest.importorskip("mistral_common.tokens.tokenizers.mistral")

TEXT = "Hello world, this is a test!"

# A name ending in a release date resolves to exactly one tokenizer, so mistral-common is
# expected to keep listing it and toko is expected to key it. Undated names are the ones
# mistral-common drops as they stop being unambiguous, so nothing is demanded of them.
DATED_RELEASE = re.compile(r"-\d{4}$")

TEKKEN_VOCAB_SIZE = 131072

MISTRAL_MODELS = next(
    group.models for group in OPTIONAL_GROUPS if group.extra == "mistral"
)


UPSTREAM_TABLE = tokenizers.MODEL_NAME_TO_TOKENIZER_CLS


def _tokenizer_file(tokenizer) -> str:
    """Which bundled file a tokenizer was loaded from, which is what identifies it.

    `file_path` is an abstract property every mistral-common tokenizer implements, and
    each bundled file appears under exactly one spec, so matching it is what makes
    "toko loads the same tokenizer upstream does" a real claim.
    """
    return Path(tokenizer.instruct_tokenizer.tokenizer.file_path).name


@pytest.mark.parametrize("model", MISTRAL_MODELS)
def test_advertised_mistral_model_counts(model):
    """Every model the mistral extra advertises counts on the installed mistral-common.

    `mistral-medium-latest` was advertised while no released mistral-common could resolve
    it, and in 1.10.0 the `-latest` names stopped resolving as a group.
    """
    counted = count_tokens(TEXT, model, use_cache=False)
    assert 5 <= counted.count <= 20


def test_rolling_names_are_approximate_and_dated_ones_are_not():
    rolling = count_tokens(TEXT, "mistral-large-latest", use_cache=False)
    assert rolling.approximate
    assert "approximate" in (rolling.caveat or "")

    dated = count_tokens(TEXT, "mistral-large-2411", use_cache=False)
    assert not dated.approximate
    assert dated.caveat is None


def test_counting_avoids_deprecated_mistral_common_api():
    """No count may go through `MistralTokenizer.from_model`.

    It warns on every version that has it and is gone in 1.13.0, so a `FutureWarning`
    here is that removal arriving early. This is the guard that holds whichever
    mistral-common the lockfile resolves.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        count_tokens(TEXT, "mistral-large-latest", use_cache=False)
        count_tokens(TEXT, "mistral-large-2411", use_cache=False)


@pytest.mark.slow
def test_mistral_specs_match_mistral_common():
    """Every shared name loads the same tokenizer file in toko as in mistral-common.

    Comparing a tokenization instead would guard almost nothing. Six of the seven specs
    give a short English sample the same count, so a count comparison misses a tekken
    model like `open-mistral-nemo-2407` mapped to v3, which counts Arabic two to three
    times high. Comparing token IDs catches that one but not `mistral-large-2411`
    mapped to v3, because v3 and v7 are the same tokenizer on text -- their vocabularies
    differ in seven control tokens no text reaches, so they part only over the chat
    template, where a system message changes the IDs without changing the count. The
    file catches both.
    """
    shared = sorted(set(MISTRAL_TOKENIZERS) & set(UPSTREAM_TABLE))
    assert shared, "mistral-common's table no longer shares a name with toko's"

    spec_file: dict[str, str] = {}
    for name in shared:
        spec = MISTRAL_TOKENIZERS[name]
        if spec not in spec_file:
            spec_file[spec] = _tokenizer_file(_load_mistral_tokenizer(spec))
        assert spec_file[spec] == _tokenizer_file(UPSTREAM_TABLE[name]()), name


@pytest.mark.slow
def test_each_spec_loads_a_distinct_tokenizer_file():
    """What makes the file a usable identity: no two specs share one."""
    specs = sorted({*MISTRAL_TOKENIZERS.values(), MISTRAL_FALLBACK_TOKENIZER})
    files = [_tokenizer_file(_load_mistral_tokenizer(spec)) for spec in specs]
    assert sorted(set(files)) == sorted(files)


def test_dated_releases_are_all_keyed():
    missing = {
        name
        for name in UPSTREAM_TABLE
        if DATED_RELEASE.search(name) and name not in MISTRAL_TOKENIZERS
    }
    assert not missing, f"dated releases would be counted approximately: {missing}"


def test_fallback_is_a_tekken_tokenizer():
    """The count an unrecognised name gets uses the vocabulary current models use."""
    tokenizer = _load_mistral_tokenizer(MISTRAL_FALLBACK_TOKENIZER)
    assert tokenizer.instruct_tokenizer.tokenizer.n_words == TEKKEN_VOCAB_SIZE
