"""Shared pytest fixtures and warning filters."""

import warnings
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
    written to whatever repository GIT_DIR names instead. Git exports GIT_DIR when
    it invokes a hook, and lefthook's pre-commit hook runs this suite, so leaving
    them set means `git commit` in a checkout rewrites that checkout's own HEAD and
    index with fixture data. That is not hypothetical; it happened.
    """
    home = tmp_path_factory.mktemp("home")
    (home / ".config").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for redirect in GIT_LOCATION_VARS:
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
