"""Tests for file_reader module."""

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pathspec
import pathspec.patterns.gitwildmatch
import pytest
import respx

import toko.file_reader
from toko.file_reader import (
    _matched_self,
    fetch_url,
    find_files,
    read_file,
    read_gitignore,
)


def test_read_file():
    """Test reading a file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, world!")
        temp_path = Path(f.name)

    try:
        content = read_file(temp_path)
        assert content == "Hello, world!"
    finally:
        temp_path.unlink()


def test_read_gitignore():
    """Test reading .gitignore file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text("*.log\n*.tmp\n")

        spec = read_gitignore(tmpdir_path)
        assert spec is not None
        assert spec.match_file("test.log")
        assert spec.match_file("test.tmp")
        assert not spec.match_file("test.txt")


def test_read_gitignore_missing():
    """Test reading .gitignore when it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = read_gitignore(Path(tmpdir))
        assert spec is None


def test_find_files_single_file():
    """Test finding a single file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test")
        temp_path = Path(f.name)

    try:
        files = find_files(temp_path)
        assert len(files) == 1
        assert files[0] == temp_path
    finally:
        temp_path.unlink()


def test_find_files_directory():
    """Test finding files in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        (tmpdir_path / "test1.txt").write_text("test1")
        (tmpdir_path / "test2.txt").write_text("test2")

        files = find_files(tmpdir_path)
        assert len(files) == 2
        assert {f.name for f in files} == {"test1.txt", "test2.txt"}


def test_find_files_recursive():
    """Test recursive directory traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create nested structure
        (tmpdir_path / "file1.txt").write_text("test1")
        subdir = tmpdir_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test2")

        # Recursive (default)
        files = find_files(tmpdir_path, recursive=True)
        assert len(files) == 2

        # Non-recursive
        files = find_files(tmpdir_path, recursive=False)
        assert len(files) == 1
        assert files[0].name == "file1.txt"


def test_find_files_gitignore():
    """Test .gitignore respect."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # A .gitignore only binds inside a repository, as it does for git and rg.
        (tmpdir_path / ".git").mkdir()

        # Create files
        (tmpdir_path / "included.txt").write_text("include")
        (tmpdir_path / "ignored.log").write_text("ignore")

        # Create .gitignore
        (tmpdir_path / ".gitignore").write_text("*.log\n")

        # With gitignore (default)
        files = find_files(tmpdir_path, respect_gitignore=True)
        assert len(files) == 1
        assert files[0].name == "included.txt"

        # Without gitignore
        files = find_files(tmpdir_path, respect_gitignore=False)
        assert len(files) == 2


def test_find_files_gitignore_directories():
    """Test .gitignore with directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / ".git").mkdir()

        # Create structure
        (tmpdir_path / "file.txt").write_text("test")
        ignored_dir = tmpdir_path / "ignored"
        ignored_dir.mkdir()
        (ignored_dir / "nested.txt").write_text("ignored")

        # Create .gitignore
        (tmpdir_path / ".gitignore").write_text("ignored/\n")

        # Should only find file.txt, not nested.txt
        files = find_files(tmpdir_path, respect_gitignore=True)
        assert len(files) == 1
        assert files[0].name == "file.txt"


def test_dot_git_is_skipped_for_being_hidden():
    """`.git` is hidden, not a boundary: `--hidden` walks it like any dotted path.

    It used to be skipped by name, and the docstring here still said so long after
    that went away. Nothing about `respect_gitignore` ever reached the hidden check,
    so parametrizing over it ran one test twice; `include_hidden` is the axis that
    tells a by-name skip apart from the dot-prefix rule.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        (tmpdir_path / "file.txt").write_text("test")
        git_dir = tmpdir_path / ".git" / "objects"
        git_dir.mkdir(parents=True)
        (tmpdir_path / ".git" / "config").write_text("[core]")
        (git_dir / "abc").write_text("object")

        assert [f.name for f in find_files(tmpdir_path)] == ["file.txt"]

        found = find_files(tmpdir_path, include_hidden=True)
        assert {str(f.relative_to(tmpdir_path)) for f in found} == {
            "file.txt",
            ".git/config",
            ".git/objects/abc",
        }


def test_find_files_exclude_patterns():
    """Test exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create files
        (tmpdir_path / "test.py").write_text("test")
        (tmpdir_path / "test.txt").write_text("test")
        (tmpdir_path / "README.md").write_text("test")

        # Exclude pattern
        files = find_files(tmpdir_path, exclude_patterns=["*.py", "*.md"])
        assert len(files) == 1
        assert files[0].name == "test.txt"


def test_find_files_not_found():
    """Test error when path doesn't exist."""
    with pytest.raises(FileNotFoundError):
        find_files(Path("/nonexistent/path"))


@respx.mock
def test_fetch_url():
    """Test fetching content from a URL."""
    url = "https://example.com/test.txt"
    content = "Hello from URL!"

    respx.get(url).mock(return_value=httpx.Response(200, text=content))

    result = fetch_url(url)
    assert result == content


@respx.mock
def test_fetch_url_404():
    """Test error when URL returns 404."""
    url = "https://example.com/notfound.txt"

    respx.get(url).mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_url(url)


@respx.mock
def test_fetch_url_redirects():
    """Test following redirects."""
    url = "https://example.com/redirect"
    final_url = "https://example.com/final"
    content = "Final content"

    # Mock redirect
    respx.get(url).mock(
        return_value=httpx.Response(302, headers={"Location": final_url})
    )
    respx.get(final_url).mock(return_value=httpx.Response(200, text=content))

    result = fetch_url(url)
    assert result == content


# Together these move the interpreter's default text encoding off UTF-8: PEP 540's
# UTF-8 mode off, and PEP 538's coercion of the C locale off too, so that LC_ALL=C is
# left meaning ASCII instead of being rescued back into UTF-8.
_NON_UTF8_ENV = {
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LANG": "C",
}
_RUN_CLI = "from toko.cli import app; app()"


def _non_utf8_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        **_NON_UTF8_ENV,
        # Kept off the developer's own config and counts, the way the CLI tests are.
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }


# The same guard the parity tests put on ripgrep, for the same reason: a skip reads as
# a pass, so CI sets this and a locale that stops taking becomes a failure there. These
# three tests cover a bug whose whole shape was counting less and exiting 0, which is
# exactly what a test that quietly stops running looks like from the outside.
REQUIRE_NON_UTF8_LOCALE = "TOKO_REQUIRE_NON_UTF8_LOCALE"


def _skip_unless_the_locale_took(env: dict[str, str]) -> None:
    got = subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import locale; print(locale.getpreferredencoding(0))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if got.lower().replace("_", "-") not in {"utf-8", "utf8"}:
        return
    if os.environ.get(REQUIRE_NON_UTF8_LOCALE):
        raise RuntimeError(
            f"{REQUIRE_NON_UTF8_LOCALE} is set but this interpreter stays on UTF-8 "
            f"({got}): the locale tests would have skipped silently, checking nothing."
        )
    pytest.skip(f"this interpreter stays on UTF-8 ({got}); nothing to prove")


def _run_toko(args: list[str], env: dict[str, str], stdin: bytes | None = None):
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _RUN_CLI, *args],
        env=env,
        input=stdin,
        capture_output=True,
        check=False,
    )


def test_a_utf8_file_is_counted_the_same_whatever_the_locale_says(tmp_path):
    """A locale-dependent decode reads a UTF-8 file as binary and drops it silently.

    The count does not merely fail, it comes back smaller and exits 0, so the caller
    has no way to tell the answer is wrong.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "ascii.txt").write_text("hello world", encoding="utf-8")
    (tree / "accented.txt").write_text("h\u00e9llo w\u00f6rld", encoding="utf-8")
    env = _non_utf8_env(tmp_path)
    _skip_unless_the_locale_took(env)

    result = _run_toko(["--format", "csv", "-m", "gpt-5", str(tree)], env)
    rows = result.stdout.decode("utf-8", "replace")

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "Skipping binary file" not in result.stderr.decode("utf-8", "replace")
    assert "ascii.txt" in rows
    assert "accented.txt" in rows


def test_utf8_piped_to_stdin_survives_a_non_utf8_locale(tmp_path):
    """Piped UTF-8 must count, whatever the locale is.

    sys.stdin decodes with the locale's encoding and surrogateescape, and the
    tokenizer then refuses the lone surrogates that leaves behind, so the run
    fails on input it should have counted.
    """
    env = _non_utf8_env(tmp_path)
    _skip_unless_the_locale_took(env)

    result = _run_toko(["-m", "gpt-5"], env, stdin="h\u00e9llo w\u00f6rld".encode())

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "surrogates not allowed" not in result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8", "replace").strip().isdigit()


def test_a_non_ascii_excludes_file_path_does_not_kill_the_run(tmp_path):
    """Git's stdout is a filesystem path, so the locale must not decide how it decodes.

    `core.excludesFile` is read by running git and reading its stdout. Decoding that
    with the locale's encoding meant a repository whose excludes file lives under a
    non-ASCII path died with UnicodeDecodeError under a non-UTF-8 locale -- taking the
    whole run with it, for a tree of plain ASCII files.
    """
    # The accent leads so that no truncated English word is left in the source for
    # the spell checker to trip over; any non-ASCII byte in the path will do.
    excludes = tmp_path / "\u00e9xcludes-dir" / "ignore"
    excludes.parent.mkdir(parents=True)
    excludes.write_text("*.swp\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(  # noqa: S603
        ["git", "config", "--global", "core.excludesFile", str(excludes)],  # noqa: S607
        cwd=repo,
        check=True,
    )
    (repo / "a.txt").write_text("hello world", encoding="utf-8")
    (repo / "notes.swp").write_text("x", encoding="utf-8")
    env = _non_utf8_env(tmp_path)
    _skip_unless_the_locale_took(env)

    result = _run_toko(["--format", "csv", "-m", "gpt-5", str(repo)], env)
    stderr = result.stderr.decode("utf-8", "replace")

    assert result.returncode == 0, stderr
    assert "codec can't decode" not in stderr
    assert "a.txt" in result.stdout.decode("utf-8", "replace")
    # The excludes file was not merely survived, it was applied.
    assert "notes.swp" not in result.stdout.decode("utf-8", "replace")


def test_a_starred_directory_exclude_still_lets_a_later_pattern_re_include(tmp_path):
    """Pruning must not swallow the directory a `dir/**` pattern only reaches inside.

    ripgrep opens `dir` for `!dir/**` and prunes what is under it, so `dir` is not
    excluded and a following `!dir/keep.txt` still has somewhere to match.
    """
    (tmp_path / "dir").mkdir()
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "dir" / "keep.txt").write_text("x")
    (tmp_path / "dir" / "drop.txt").write_text("x")

    found = find_files(tmp_path, exclude_patterns=["dir/**", "!dir/keep.txt"])

    assert {path.name for path in found} == {"top.txt", "keep.txt"}


def test_a_bare_directory_exclude_leaves_a_later_negation_nothing_to_re_include(
    tmp_path,
):
    """Deliberate parity with ripgrep and git, and the exact counterpart of the test above.

    `dir/**` reaches only what is inside `dir`, so the walk still opens `dir` and the
    negation lands. `dir/` excludes the directory itself, the walk prunes it, and the
    negation has nothing left to reach -- neither ripgrep nor gitignore can re-include a
    file whose parent directory is excluded. `keep.txt` going missing here is therefore
    the intended outcome; restoring it would put toko back to pruning directories
    ripgrep descends into. Change this expectation only by changing that decision.
    """
    (tmp_path / "dir").mkdir()
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "dir" / "keep.txt").write_text("x")
    (tmp_path / "dir" / "drop.txt").write_text("x")

    found = find_files(tmp_path, exclude_patterns=["dir/", "!dir/keep.txt"])

    assert {path.name for path in found} == {"top.txt"}


@pytest.mark.parametrize("rule", ["dir/", "dir/ ", "dir/\t", "dir/ \t "])
def test_a_directory_exclude_with_trailing_whitespace_still_excludes(tmp_path, rule):
    """Pathspec strips the rule before compiling it, so the reader has to strip it too.

    `dir/ ` and `dir/` compile to one regex. Judging directory-only-ness from the raw
    text instead calls the first a file rule, probes `dir` against a regex that wants
    `dir/`, and lets the walk into a directory the user excluded.
    """
    (tmp_path / "dir").mkdir()
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "dir" / "secret.txt").write_text("x")

    found = find_files(tmp_path, exclude_patterns=[rule])

    assert {path.name for path in found} == {"top.txt"}


def test_an_escaped_trailing_space_is_not_stripped_into_a_directory_rule(tmp_path):
    r"""The counterpart: `dir/\ ` names a file called " " and must stay a file rule.

    pathspec strips both ends of a pattern unless it ends in a backslash-escaped
    space, and stripping that one anyway would turn a rule about a file into a rule
    that prunes a directory. So the stripping has to stop where pathspec's stops.
    """
    (tmp_path / "dir").mkdir()
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "dir" / "keep.txt").write_text("x")

    found = find_files(tmp_path, exclude_patterns=["dir/\\ "])

    assert {path.name for path in found} == {"top.txt", "keep.txt"}


@pytest.fixture
def pathspec_without_the_descendant_mark(monkeypatch):
    """Stand in for a pathspec release that renames the private group toko reads.

    Nothing in pathspec's public surface promises `ps_d`, and the requirement carries
    no upper bound, so this is the shape of the regression the import-time guard exists
    to turn into an exception.
    """
    original = pathspec.patterns.gitwildmatch.GitWildMatchPattern.pattern_to_regex

    def renamed(pattern):
        regex, include = original(pattern)
        if regex is not None:
            regex = regex.replace("?P<ps_d>", "?P<ps_descendant>")
        return regex, include

    monkeypatch.setattr(
        pathspec.patterns.gitwildmatch.GitWildMatchPattern,
        "pattern_to_regex",
        staticmethod(renamed),
    )
    yield
    # A reload that raised part way through leaves the module object stripped of every
    # name defined below the guard, so rebuild it once pathspec is itself again.
    monkeypatch.undo()
    importlib.reload(toko.file_reader)


@pytest.mark.usefixtures("pathspec_without_the_descendant_mark")
def test_losing_the_pathspec_descendant_mark_silently_restores_pruning():
    """Why the guard is worth having: the regression it catches raises nothing itself."""
    pattern = next(
        iter(pathspec.PathSpec.from_lines("gitwildmatch", ["dir/"]).patterns)
    )

    # `dir/below.txt` is something *inside* `dir`, so the honest verdict is None and
    # the walk is left to prune. Without the mark it reads as the directory's own
    # match, and a later `!dir/keep.txt` loses the directory it needed opened.
    assert _matched_self(pattern, "dir/below.txt") is True


@pytest.mark.usefixtures("pathspec_without_the_descendant_mark")
def test_importing_fails_loudly_when_pathspec_drops_the_descendant_mark():
    """The same regression met at import: an exception naming what broke, not silence."""
    with pytest.raises(RuntimeError, match="pathspec") as raised:
        importlib.reload(toko.file_reader)

    assert "ps_d" in str(raised.value)


@pytest.fixture
def pathspec_with_a_widened_descendant_mark(monkeypatch):
    """Stand in for a pathspec release that keeps the group name but grows its reach.

    The harder half of the same regression: a guard that only asks whether `ps_d`
    exists still passes here, because it does exist. What breaks is how far it
    reaches, which is the property `_matched_self` actually reads.
    """
    original = pathspec.patterns.gitwildmatch.GitWildMatchPattern.pattern_to_regex

    def widened(pattern):
        regex, include = original(pattern)
        if regex is not None:
            regex = regex.replace("(?P<ps_d>/)", "(?P<ps_d>/.*)")
        return regex, include

    monkeypatch.setattr(
        pathspec.patterns.gitwildmatch.GitWildMatchPattern,
        "pattern_to_regex",
        staticmethod(widened),
    )
    yield
    monkeypatch.undo()
    importlib.reload(toko.file_reader)


@pytest.mark.usefixtures("pathspec_with_a_widened_descendant_mark")
def test_a_widened_descendant_mark_silently_restores_pruning():
    """Why the name alone is not enough to check: the group is still there, still named."""
    pattern = next(
        iter(pathspec.PathSpec.from_lines("gitwildmatch", ["dir/"]).patterns)
    )

    assert pattern.regex.groupindex.get("ps_d") is not None
    # Widened, the mark now reaches the end of every descendant path, so the test that
    # tells a directory's own match apart -- how far the mark reaches -- reads True for
    # something inside `dir` just as it does for `dir` itself.
    assert _matched_self(pattern, "dir/below.txt") is True


@pytest.mark.usefixtures("pathspec_with_a_widened_descendant_mark")
def test_importing_fails_loudly_when_pathspec_widens_the_descendant_mark():
    """The guard has to exercise the semantics, not just look the name up."""
    with pytest.raises(RuntimeError, match="pathspec") as raised:
        importlib.reload(toko.file_reader)

    assert "ps_d" in str(raised.value)
