"""Differential test: the files toko discovers are the files `rg --files` lists."""

import shutil
import subprocess
from pathlib import Path  # noqa: TC003

import pytest

from toko.file_reader import find_files

pytestmark = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep is not installed"
)

RIPGREP = "rg"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)  # noqa: S603, S607


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run_ripgrep(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603
        [RIPGREP, "--files", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    # 2 is ripgrep's "something went wrong", which --follow returns for this tree's
    # dangling link and its cycle while still listing every file it did reach.
    assert result.returncode in (0, 2), result.stderr
    return result


def ripgrep_files(root: Path, *args: str) -> set[str]:
    return set(run_ripgrep(root, *args).stdout.split("\n")) - {""}


def toko_files(root: Path, *, hidden: bool, follow: bool = False) -> set[str]:
    return {
        str(f.absolute().relative_to(root.absolute()))
        for f in find_files(root, include_hidden=hidden, follow_symlinks=follow)
    }


@pytest.fixture
def parity_tree(tmp_path, isolated_git_env, monkeypatch) -> Path:
    """Every ignore source ripgrep honours, each with a file only it can exclude."""
    # ripgrep never reads GIT_CONFIG_GLOBAL; it looks for $HOME/.gitconfig and
    # $XDG_CONFIG_HOME/git/config itself. Pointing git at $HOME/.gitconfig is the
    # one place both of them agree to read.
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    excludes = write(isolated_git_env / "global-excludes", "*.swp\n")
    write(gitconfig, f"[core]\n\texcludesFile = {excludes}\n")

    write(tmp_path / ".gitignore", "*.above-repo\n")
    write(tmp_path / ".ignore", "*.above-repo-dot\n")

    repo = tmp_path / "fx"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    write(repo / ".gitignore", "*.log\n")
    write(repo / ".git" / "info" / "exclude", "excluded-by-info.txt\n")
    write(repo / ".ignore", "dot-ignored.txt\n")
    write(repo / ".rgignore", "rg-ignored.txt\n")

    for name in (
        "keep.txt",
        "top.log",
        "excluded-by-info.txt",
        "notes.swp",
        "dot-ignored.txt",
        "rg-ignored.txt",
        ".hidden.txt",
        "kept.above-repo",
        "dropped.above-repo-dot",
    ):
        write(repo / name)
    (repo / "binary.bin").write_bytes(b"\x00\x01\x02 not text")
    write(repo / ".hiddendir" / "inside.txt")

    write(repo / "sub" / ".gitignore", "nested-ignored.txt\n")
    for name in ("keep.txt", "nested-ignored.txt", "top.log"):
        write(repo / "sub" / name)

    nested = repo / "vendor" / "lib"
    nested.mkdir(parents=True)
    git("init", "-q", cwd=nested)
    write(nested / ".gitignore", "target/\n")
    write(nested / ".git" / "info" / "exclude", "vendored.tmp\n")
    write(nested / "src.rs")
    write(nested / "vendored.tmp")
    write(nested / "target" / "build.rs")

    # Every shape --follow has to cope with: a link to a file, a link to a directory,
    # one that resolves to nothing, and one that points back at an ancestor.
    (repo / "link.txt").symlink_to(repo / "keep.txt")
    (repo / "linkdir").symlink_to(repo / "sub")
    (repo / "dangling.txt").symlink_to(repo / "gone.txt")
    write(repo / "cycle" / "a.txt")
    (repo / "cycle" / "up").symlink_to("..")

    # A commit fills .git with loose objects, refs and logs, so --hidden parity
    # is measured against a real repository rather than a bare skeleton.
    git("add", "--", ".gitignore", "keep.txt", "sub", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "fx", cwd=repo)
    return repo


@pytest.mark.parametrize("subpath", ["", "sub"])
@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("hidden", [False, True])
def test_discovery_matches_ripgrep(parity_tree, subpath, hidden, follow):
    root = parity_tree / subpath if subpath else parity_tree
    args = [*(["--hidden"] if hidden else []), *(["--follow"] if follow else [])]

    # `rg --files` lists binary files; toko warns about them later, when it reads
    # them. Comparing here compares discovery, which is what parity is about.
    assert toko_files(root, hidden=hidden, follow=follow) == ripgrep_files(root, *args)


def test_the_tree_actually_exercises_every_source(parity_tree):
    found = ripgrep_files(parity_tree)
    assert found == {
        "binary.bin",
        "cycle/a.txt",
        "keep.txt",
        "kept.above-repo",
        "sub/keep.txt",
        "vendor/lib/src.rs",
    }


def test_the_tree_has_a_populated_git_directory_for_the_hidden_case(parity_tree):
    found = ripgrep_files(parity_tree, "--hidden")
    assert {".git/HEAD", ".git/config", ".git/index"} <= found
    assert [name for name in found if name.startswith(".git/objects/")]


def test_the_tree_makes_ripgrep_report_both_a_dangling_link_and_a_loop(parity_tree):
    """The --follow parity case is only worth anything if the tree provokes both."""
    stderr = run_ripgrep(parity_tree, "--follow").stderr

    assert "No such file or directory" in stderr
    assert "File system loop found" in stderr


def test_following_is_what_brings_the_symlinked_paths_back(parity_tree):
    without = toko_files(parity_tree, hidden=False)
    following = toko_files(parity_tree, hidden=False, follow=True)

    assert following - without == {"link.txt", "linkdir/keep.txt"}


def test_a_dangling_link_and_a_loop_are_reported_rather_than_raised(parity_tree):
    problems: list[str] = []

    found = find_files(parity_tree, follow_symlinks=True, on_error=problems.append)

    assert any(
        "dangling.txt" in problem and "No such file" in problem for problem in problems
    )
    assert any(
        problem.startswith("File system loop found") and "cycle/up" in problem
        for problem in problems
    )
    # The rest of the tree is still counted: a bad link costs its own path, not the run.
    assert "keep.txt" in {path.name for path in found}
