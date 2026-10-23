"""Shared pytest fixtures and warning filters."""

import os
import warnings
from collections.abc import Mapping  # noqa: TC003
from pathlib import Path  # noqa: TC003

import pytest
from genai_prices.data_snapshot import set_custom_snapshot

import toko.counter as counter
from toko.cache import clear_cache, set_cache_dir

# Suppress noisy DeprecationWarnings emitted by optional tokenizer backends.
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


# Every variable that redirects git away from directory-based discovery. Anything
# that can point git at a different repository, index or object store belongs here.
GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    "GIT_CEILING_DIRECTORIES",
)

# Config that git reads from the environment instead of from a file. These bypass
# both GIT_CONFIG_NOSYSTEM and GIT_CONFIG_GLOBAL, so redirecting those two is not
# enough: this environment exports GIT_CONFIG_COUNT=3 to hooks, and with it set a
# fixture git sees the ambient `url.*.insteadOf` rewrites.
GIT_CONFIG_VARS = ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS")

# GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n are numbered up to GIT_CONFIG_COUNT, so they
# cannot be listed; they have to be matched. Dropping the count without them would
# leave git erroring on the orphans, so all three go together.
GIT_CONFIG_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")


def git_vars_to_scrub(environ: Mapping[str, str]) -> set[str]:
    numbered = {name for name in environ if name.startswith(GIT_CONFIG_PREFIXES)}
    return {*GIT_LOCATION_VARS, *GIT_CONFIG_VARS, *numbered}


@pytest.fixture(autouse=True)
def isolated_git_env(monkeypatch, tmp_path_factory) -> Path:
    """Keep every test off the developer's real git config and home directory.

    Autouse rather than opt-in: file discovery reads the ambient `core.excludesFile`
    from any test that walks a directory, so a module that forgets to ask for this
    passes or fails according to whose machine it runs on. The home is made outside
    `tmp_path` so that a test asserting on the exact contents of its own `tmp_path`
    does not have to know this fixture exists.

    GIT_LOCATION_VARS are cleared because several tests build fixtures by running
    `git init` and `git commit` inside `tmp_path`, and those variables override
    directory-based discovery: with GIT_DIR set, a commit made in a fixture is
    written to whatever repository GIT_DIR names instead. Git exports GIT_DIR to a
    hook only when the git dir is not the `.git` beside the working tree -- that is,
    from a LINKED WORKTREE, and not from an ordinary clone. That qualifier is the
    whole of it: reproducing this from a plain checkout shows nothing and reads as a
    disproof. From a worktree, with lefthook's pre-commit hook running this suite,
    a leaked GIT_DIR sends every fixture `git commit` into the checkout being
    committed to and rewrites its HEAD and index. That is not hypothetical; it
    happened, and GIT_DIR alone was later shown to be enough to cause it.
    """
    home = tmp_path_factory.mktemp("home")
    (home / ".config").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for redirect in git_vars_to_scrub(os.environ):
        monkeypatch.delenv(redirect, raising=False)
    return home


@pytest.fixture(autouse=True)
def _cache_root(tmp_path):
    cache_root = tmp_path / "cache"
    set_cache_dir(cache_root)
    try:
        yield cache_root
    finally:
        clear_cache()
        set_cache_dir(None)


@pytest.fixture(autouse=True)
def _bundled_prices():
    """Keep fetched prices from leaking between tests."""
    try:
        yield
    finally:
        set_custom_snapshot(None)


@pytest.fixture
def cache_dir(_cache_root):
    return _cache_root


@pytest.fixture(autouse=True)
def _reset_warned_models():
    """Start every test from an empty warned-once registry.

    It is process-global, so a "warns once" assertion is otherwise order-dependent.
    """
    counter._WARNED_ONCE.clear()  # noqa: SLF001
    yield
    counter._WARNED_ONCE.clear()  # noqa: SLF001
