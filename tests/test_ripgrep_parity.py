"""Differential test: the files toko discovers are the files `rg --files` lists."""

import contextlib
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from tests.git_runner import run_git
from toko.file_reader import find_files

RIPGREP = "rg"

# A skipped module reads as a pass, so CI sets TOKO_REQUIRE_RG and a missing ripgrep
# becomes a collection error there; locally the tests still just skip.
if shutil.which(RIPGREP) is None:
    if os.environ.get("TOKO_REQUIRE_RG"):
        raise RuntimeError(
            f"TOKO_REQUIRE_RG is set but {RIPGREP!r} is not on PATH: the parity tests "
            "would have skipped silently, checking nothing."
        )
    pytestmark = pytest.mark.skip(reason="ripgrep is not installed")


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def spelled_from(root: Path, cwd: Path) -> str:
    """How the walk root is named on a command line run from `cwd`."""
    return os.path.relpath(root, cwd)


def run_ripgrep(
    root: Path, *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """List `root` with ripgrep, from `cwd` when the walk is not started inside it.

    Until this took a `cwd`, every fixture in this file ran ripgrep with its working
    directory equal to the walk root, so the suite was structurally blind to every
    rule ripgrep resolves against the working directory rather than against the tree
    -- core.excludesFile is one. That was a property of the harness, not of any one
    test: no fixture could have caught it, however many were added.
    """
    target = [] if cwd is None else ["--", spelled_from(root, cwd)]
    result = subprocess.run(  # noqa: S603
        [RIPGREP, "--files", *args, *target],
        cwd=root if cwd is None else cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    # 2 is ripgrep's "something went wrong", which --follow returns for this tree's
    # dangling link and its cycle while still listing every file it did reach.
    assert result.returncode in (0, 2), result.stderr
    return result


def ripgrep_files(root: Path, *args: str, cwd: Path | None = None) -> set[str]:
    listed = set(run_ripgrep(root, *args, cwd=cwd).stdout.split("\n")) - {""}
    if cwd is None:
        return listed
    # ripgrep echoes the path it was given, so its listing is relative to `cwd`; the
    # comparison is against toko's, which is relative to the walk root.
    prefix = f"{spelled_from(root, cwd)}/"
    return {name.removeprefix(prefix) for name in listed}


def toko_files(
    root: Path, *, hidden: bool, follow: bool = False, cwd: Path | None = None
) -> set[str]:
    """Walk with toko from the same working directory ripgrep was given.

    The working directory is set rather than inherited because ripgrep's is: leaving
    toko in whatever directory pytest was started from would compare two walks that
    disagree about where the process is standing.
    """
    base = root.absolute()
    workdir = root if cwd is None else cwd
    with contextlib.chdir(workdir):
        # Named the way ripgrep is given it, because ripgrep resolves the global
        # excludes file against the working directory and reads each path as spelled:
        # handing toko an absolute root while ripgrep gets a relative one compares two
        # walks that were asked different questions.
        found = find_files(
            Path(spelled_from(root, workdir)),
            include_hidden=hidden,
            follow_symlinks=follow,
        )
        # abspath, not Path.absolute: a walk root spelled `..` leaves the `..` in
        # every result, and relative_to compares the text rather than the directory.
        return {
            str(Path(os.path.abspath(f)).relative_to(base))  # noqa: PTH100
            for f in found
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
    run_git(repo, "init", "-q")
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
    run_git(nested, "init", "-q")
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
    run_git(repo, "add", "--", ".gitignore", "keep.txt", "sub")
    run_git(repo, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "fx")
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


@pytest.fixture
def ignored_loop_tree(tmp_path) -> Path:
    """Build a tree whose symlink loop sits on an entry an ignore rule prunes.

    The parity tree's cycle is somewhere no ignore rule reaches, so it cannot tell
    whether loop detection runs before or after the ignore check. `kept/loop` is a
    real directory the same rule prunes, so a listing that drops it proves the rule
    is live -- otherwise a loop reported here would only mean the rule never applied.
    """
    write(tmp_path / "kept" / "a.txt")
    write(tmp_path / "kept" / "loop" / "c.txt")
    write(tmp_path / "pruned" / "b.txt")
    (tmp_path / "pruned" / "loop").symlink_to(tmp_path)
    write(tmp_path / ".ignore", "loop\n")
    return tmp_path


def test_a_loop_through_an_ignored_directory_is_reported_the_way_ripgrep_reports_it(
    ignored_loop_tree,
):
    """Ripgrep reports a loop it has no intention of descending; toko exited 0."""
    result = run_ripgrep(ignored_loop_tree, "--follow")
    problems: list[str] = []

    found = find_files(
        ignored_loop_tree, follow_symlinks=True, on_error=problems.append
    )

    assert "File system loop found" in result.stderr
    assert result.returncode == 2
    assert any(problem.startswith("File system loop found") for problem in problems)
    listed = {
        str(path.absolute().relative_to(ignored_loop_tree.absolute())) for path in found
    }
    assert listed == ripgrep_files(ignored_loop_tree, "--follow")
    assert "kept/loop/c.txt" not in listed


@pytest.fixture
def exclude_tree(tmp_path) -> Path:
    """Build a tree that tells a pruning pattern apart from a filtering one.

    `dir` holds both a file and a subdirectory, so `dir/**` and `dir/` differ in what
    ripgrep opens even though they exclude the same files. `notes.md` keeps a pattern
    as broad as `*.txt` from emptying the listing, which ripgrep exits 1 for. `other/dir`
    is a plain file with a directory's name, the one path `dir/` and `dir` disagree on.
    """
    for name in (
        "top.txt",
        "notes.md",
        "dir/a.txt",
        "dir/deep/b.txt",
        "other/c.txt",
        "other/dir",
    ):
        write(tmp_path / name)
    return tmp_path


def toko_scanned(root: Path, **kwargs) -> set[str]:
    """Collect the directories a walk actually opened, relative to its root."""
    opened: list[Path] = []
    real_scandir = os.scandir

    def spy(path):
        opened.append(Path(path).absolute())
        return real_scandir(path)

    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(os, "scandir", spy)
        find_files(root, **kwargs)
    finally:
        patch.undo()
    return {str(path.relative_to(root.absolute())) for path in opened}


def ripgrep_skipped_dirs(root: Path, *args: str) -> set[str]:
    """Read off the directories ripgrep's debug log says it refused to descend into."""
    stderr = run_ripgrep(root, "--debug", *args).stderr
    skipped = (
        line.split("ignoring ", 1)[1].split(":", 1)[0].removeprefix("./")
        for line in stderr.splitlines()
        if "ignoring " in line
    )
    return {path for path in skipped if (root / path).is_dir()}


def all_dirs(root: Path) -> set[str]:
    return {".", *(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())}


def _lineage(directory: str) -> set[str]:
    """Name a directory and every directory above it, up to the walk's root."""
    path = PurePosixPath(directory)
    return {str(path), *(str(parent) for parent in path.parents)}


def ripgrep_descended_dirs(root: Path, *args: str) -> set[str]:
    """Invert ripgrep's prune log into the set of directories it opened.

    `--debug` names the directory a rule pruned and nothing under it, because nothing
    under it is ever reached, so the complement has to drop a pruned directory's
    descendants as well as the directory itself.
    """
    skipped = ripgrep_skipped_dirs(root, *args)
    return {name for name in all_dirs(root) if not _lineage(name) & skipped}


# Eight patterns; three distinct file listings between them. What separates the
# eight is the directory each one opens, so the scan set is the observable that
# earns the parametrization -- the file list is the same for most of them.
EXCLUDE_PATTERN_SCANS = [
    ("dir", {".", "other"}),
    ("dir/", {".", "other"}),
    ("dir/*", {".", "dir", "other"}),
    ("dir/**", {".", "dir", "other"}),
    ("**/dir/**", {".", "dir", "other"}),
    ("dir/deep", {".", "dir", "other"}),
    ("dir/deep/**", {".", "dir", "dir/deep", "other"}),
    ("*.txt", {".", "dir", "dir/deep", "other"}),
]


@pytest.mark.parametrize("pattern", [pattern for pattern, _ in EXCLUDE_PATTERN_SCANS])
def test_an_exclude_pattern_drops_what_a_negated_ripgrep_glob_drops(
    exclude_tree, pattern
):
    found = {
        str(path.absolute().relative_to(exclude_tree.absolute()))
        for path in find_files(exclude_tree, exclude_patterns=[pattern])
    }

    assert found == ripgrep_files(exclude_tree, "-g", f"!{pattern}")


@pytest.mark.parametrize(("pattern", "scanned"), EXCLUDE_PATTERN_SCANS)
def test_an_exclude_pattern_prunes_exactly_the_directories_ripgrep_prunes(
    exclude_tree, pattern, scanned
):
    """`dir/` prunes `dir`; `dir/**` does not, because it only matches what is inside.

    The two exclude the same files, and so do most of the eight, so the file lists
    cannot tell them apart -- which is why the sibling test above is not enough on its
    own. What the walk opens can, and it is the whole point of the flag: matching
    `dir/**` against `dir/` would prune a directory ripgrep descends into.
    """
    opened = toko_scanned(exclude_tree, exclude_patterns=[pattern])

    assert opened == scanned
    assert opened == ripgrep_descended_dirs(exclude_tree, "-g", f"!{pattern}")


@pytest.mark.parametrize(
    ("pattern", "keeps_the_file"), [("dir/", True), ("dir", False)]
)
def test_a_directory_only_exclude_spares_a_file_that_shares_the_name(
    exclude_tree, pattern, keeps_the_file
):
    """`dir/` matches only directories; `dir` matches a file called `dir` as well.

    The two prune the same directory and drop the same files under it, so `other/dir`
    is the only path in the tree that tells them apart. Without it, dropping the
    trailing separator from every exclude pattern leaves the whole suite green.
    """
    found = {
        str(path.absolute().relative_to(exclude_tree.absolute()))
        for path in find_files(exclude_tree, exclude_patterns=[pattern])
    }

    assert found == ripgrep_files(exclude_tree, "-g", f"!{pattern}")
    assert ("other/dir" in found) is keeps_the_file


@pytest.fixture
def starred_tree(tmp_path) -> Path:
    """Build the one tree where pruning `dir` and filtering inside it disagree.

    `dir/**` reaches only what is under `dir`, so ripgrep opens `dir` and a following
    `!dir/keep.txt` still has somewhere to match. Prune `dir` instead and the listings
    stay identical except for that single file, which is why probing every rule with a
    trailing separator dropped it from the count without anything looking wrong.
    """
    run_git(tmp_path, "init", "-q")
    for name in (
        "top.txt",
        "dir/keep.txt",
        "dir/drop.txt",
        "dir/deep/b.txt",
        "other/c.txt",
    ):
        write(tmp_path / name)
    return tmp_path


@pytest.mark.parametrize(
    "source", [".gitignore", ".ignore", ".rgignore", ".git/info/exclude"]
)
def test_a_starred_directory_rule_leaves_a_later_negation_somewhere_to_match(
    starred_tree, source
):
    """Each ignore file is a separate source reaching one probe; none may prune `dir`."""
    write(starred_tree / source, "dir/**\n!dir/keep.txt\n")

    found = toko_files(starred_tree, hidden=False)

    assert found == ripgrep_files(starred_tree)
    assert "dir/keep.txt" in found


def test_a_bare_directory_rule_leaves_a_later_negation_nothing_to_re_include(
    starred_tree,
):
    """Ripgrep parity, deliberately: an excluded directory is pruned and never reopened.

    The contrast with the starred test above is the whole point of keeping both. `dir/**`
    matches only what is inside `dir`, so `dir` stays open and `!dir/keep.txt` lands;
    `dir/` excludes the directory itself, so both walks prune it and the negation has
    nothing left to reach. Neither ripgrep nor git can re-include a file under an
    excluded directory, and toko losing `dir/keep.txt` here is that same rule, measured
    against real ripgrep -- not an oversight, and not something to "fix" back.
    """
    write(starred_tree / ".gitignore", "dir/\n!dir/keep.txt\n")

    found = toko_files(starred_tree, hidden=False)

    assert found == ripgrep_files(starred_tree)
    assert "dir/keep.txt" not in found
    # And for the same reason in both: the directory is pruned, not filtered.
    assert "dir" in ripgrep_skipped_dirs(starred_tree)
    assert toko_scanned(starred_tree) == {".", "other"}


def test_a_starred_rule_in_the_global_excludes_file_also_spares_the_negation(
    starred_tree, isolated_git_env, monkeypatch
):
    """The one ignore source that is not a file in the tree, held to the same rule."""
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    excludes = write(isolated_git_env / "global-excludes", "dir/**\n!dir/keep.txt\n")
    write(gitconfig, f"[core]\n\texcludesFile = {excludes}\n")

    found = toko_files(starred_tree, hidden=False)

    assert found == ripgrep_files(starred_tree)
    assert "dir/keep.txt" in found


@pytest.fixture
def anchored_excludes_tree(tmp_path, isolated_git_env, monkeypatch) -> Path:
    """Build a tree whose global excludes file needs a directory to be resolved against.

    `*.swp`, the pattern the other fixtures use, matches on its own wherever it is
    read from, so it cannot show where the file is anchored. `/x.txt` is anchored and
    `sub/y.txt` spans two segments; both need a directory to be resolved against, and
    ripgrep resolves them against the process's working directory.
    """
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    excludes = write(isolated_git_env / "anchored-excludes", "/x.txt\nsub/y.txt\n")
    write(gitconfig, f"[core]\n\texcludesFile = {excludes}\n")

    run_git(tmp_path, "init", "-q")
    for name in (
        "x.txt",
        "keep.txt",
        "sub/x.txt",
        "sub/y.txt",
        "sub/keep.txt",
        "sub/deep/y.txt",
    ):
        write(tmp_path / name)
    return tmp_path


@pytest.mark.parametrize("workdir", ["", "sub", "outside"])
@pytest.mark.parametrize("walk", ["", "sub"])
def test_the_global_excludes_file_is_anchored_at_the_working_directory(
    anchored_excludes_tree, tmp_path_factory, walk, workdir
):
    """Every combination of where the walk starts and where the process is standing.

    `workdir="sub"` is the one that reads the same excludes file to a different answer
    -- `/x.txt` then names `sub/x.txt` and `sub/y.txt` names nothing -- and `outside`
    is the case where nothing anchored can be resolved at all. git would answer all
    three from the repository root; ripgrep answers them from the working directory,
    and so does toko -- deliberately, and recorded as issue 133.
    """
    root = anchored_excludes_tree / walk if walk else anchored_excludes_tree
    if workdir == "outside":
        cwd = tmp_path_factory.mktemp("elsewhere")
    else:
        cwd = anchored_excludes_tree / workdir if workdir else anchored_excludes_tree

    assert toko_files(root, hidden=False, cwd=cwd) == ripgrep_files(root, cwd=cwd)


def test_the_anchored_excludes_tree_reads_differently_from_different_directories(
    anchored_excludes_tree,
):
    """The fixture is only worth anything if ripgrep's own answer moves with its cwd."""
    root = anchored_excludes_tree
    sub = root / "sub"

    assert ripgrep_files(root, cwd=root) == {
        "keep.txt",
        "sub/x.txt",
        "sub/keep.txt",
        "sub/deep/y.txt",
    }
    assert ripgrep_files(sub, cwd=sub) == {"y.txt", "keep.txt", "deep/y.txt"}
    # From the repository root the same walk of `sub` drops y.txt and keeps x.txt,
    # which is the exact reverse of the listing above.
    assert ripgrep_files(sub, cwd=root) == {"x.txt", "keep.txt", "deep/y.txt"}


def test_an_ignore_file_prunes_exactly_the_directories_ripgrep_prunes(starred_tree):
    """The mechanism under the re-include: `dir` is opened, `dir/deep` is not."""
    write(starred_tree / ".gitignore", "dir/**\n!dir/keep.txt\n")
    skipped = ripgrep_skipped_dirs(starred_tree)

    assert "dir/deep" in skipped
    assert "dir" not in skipped

    assert toko_scanned(starred_tree) == {".", "dir", "other"}


def test_a_gitignored_directory_is_one_ripgrep_refuses_to_descend_into(tmp_path):
    """A pruned directory is never opened, measured on the primitive the walk uses.

    The version of this that spied on `os.walk` recorded nothing, because the walk has
    always used `os.scandir`; its "nothing ignored was visited" assertion then held
    over an empty list. Asserting the spy saw the root keeps that failure loud.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "node_modules/\n")
    write(tmp_path / "app.js")
    for index in range(5):
        write(tmp_path / "node_modules" / f"pkg{index}" / "index.js")

    scanned = toko_scanned(tmp_path)
    skipped = ripgrep_skipped_dirs(tmp_path)

    assert "." in scanned, "the scandir spy recorded nothing, so it proves nothing"
    assert "node_modules" in skipped
    assert scanned.isdisjoint(skipped)
    assert scanned == {"."}


@pytest.mark.parametrize(
    "rule", ["dir/", "dir/ ", "dir/  ", "dir/\t", "dir/ \t ", "dir "]
)
def test_trailing_whitespace_leaves_a_directory_rule_still_pruning(tmp_path, rule):
    """Git and ripgrep both strip a rule's trailing whitespace; so must whatever reads it.

    pathspec compiles `dir/ ` to the regex it compiles `dir/` to, so a reader that asks
    the raw text whether the rule is directory-only calls this one a file rule, probes
    `dir` where the regex wants `dir/`, matches nothing and descends -- handing the
    contents of an excluded directory to a counter that sends them to a provider.
    The whitespace spellings are invisible in an editor, which is what makes it quiet.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", f"{rule}\n")
    write(tmp_path / "top.txt")
    write(tmp_path / "dir" / "a.txt")

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)
    # The tree proves something only because the rule has to bite for this to hold.
    assert ripgrep_files(tmp_path) == {"top.txt"}


def test_a_directory_rule_with_trailing_whitespace_is_never_descended_into(tmp_path):
    """The listing can come out right while the pruning is lost; measure the walk itself.

    A rule that stops pruning still filters each file it finds, so a tree whose every
    ignored path is a plain file cannot tell the two apart. Watching which directories
    are opened is what separates them.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "dir/ \n")
    write(tmp_path / "top.txt")
    for index in range(3):
        write(tmp_path / "dir" / f"pkg{index}" / "index.js")

    scanned = toko_scanned(tmp_path)
    skipped = ripgrep_skipped_dirs(tmp_path)

    assert "." in scanned, "the scandir spy recorded nothing, so it proves nothing"
    assert "dir" in skipped
    assert scanned.isdisjoint(skipped)
    assert scanned == {"."}


def test_a_root_rgignore_beats_a_deeper_ignore(tmp_path):
    """Rank comes before depth: the kinds are separate tiers, not one ordered list."""
    write(tmp_path / ".rgignore", "shadowed.txt\n")
    write(tmp_path / "sub" / ".ignore", "!shadowed.txt\n")
    write(tmp_path / "sub" / "shadowed.txt")
    write(tmp_path / "sub" / "kept.txt")

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)
    # The tree proves something only because the two files disagree about this path.
    assert ripgrep_files(tmp_path) == {"sub/kept.txt"}


def test_a_deeper_rgignore_still_beats_a_root_ignore(tmp_path):
    """The same ranking read the other way, so neither order alone can pass both."""
    write(tmp_path / ".ignore", "shadowed.txt\n")
    write(tmp_path / "sub" / ".rgignore", "!shadowed.txt\n")
    write(tmp_path / "sub" / "shadowed.txt")
    write(tmp_path / "shadowed.txt")

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)
    assert ripgrep_files(tmp_path) == {"sub/shadowed.txt"}


@pytest.fixture
def excludes_file_repo(tmp_path, isolated_git_env, monkeypatch) -> Path:
    """Build a repository whose own core.excludesFile disagrees with the global one."""
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    global_excludes = write(isolated_git_env / "global-excludes", "*.dropped\n")
    write(gitconfig, f"[core]\n\texcludesFile = {global_excludes}\n")
    write(isolated_git_env / "repo-excludes", "*.kept\n")

    repo = tmp_path / "fx"
    repo.mkdir()
    run_git(repo, "init", "-q")
    for name in ("keep.txt", "one.dropped", "two.kept"):
        write(repo / name)
    return repo


def test_a_repository_local_excludes_file_is_passed_over_as_ripgrep_passes_it_over(
    excludes_file_repo, isolated_git_env
):
    """Ripgrep parses the global git config itself and never opens a repository's own.

    `git status` in this repository hides the opposite file, so reading the setting
    with a plain `git config --get` -- which answers from the repository first --
    excluded a file ripgrep lists and listed one ripgrep excludes.
    """
    run_git(
        excludes_file_repo,
        "config",
        "--local",
        "core.excludesFile",
        str(isolated_git_env / "repo-excludes"),
    )

    assert toko_files(excludes_file_repo, hidden=False) == ripgrep_files(
        excludes_file_repo
    )
    assert ripgrep_files(excludes_file_repo) == {"keep.txt", "two.kept"}


def test_a_repository_local_excludes_file_naming_nothing_leaves_the_global_one_alone(
    excludes_file_repo,
):
    """Point the local setting at nothing, and ripgrep still applies the global file.

    For git, a local setting naming a path that does not exist suppresses the global
    one and the excluded paths come back. ripgrep never reads the setting at all.
    """
    run_git(
        excludes_file_repo,
        "config",
        "--local",
        "core.excludesFile",
        str(excludes_file_repo / "no-such-file"),
    )

    assert toko_files(excludes_file_repo, hidden=False) == ripgrep_files(
        excludes_file_repo
    )
    assert ripgrep_files(excludes_file_repo) == {"keep.txt", "two.kept"}


def listed_by_toko(root: Path, argument: str) -> set[str]:
    """List what toko finds for one named argument, spelled relative to `root`."""
    return {
        str(path.absolute().relative_to(root.absolute()))
        for path in find_files(root / argument)
    }


def listed_by_ripgrep(root: Path, argument: str) -> set[str]:
    """Ask ripgrep for the same listing, re-spelled so the two are comparable.

    ripgrep echoes each argument's own spelling back in its output, so a bare, a
    dot-relative and an absolute argument produce three different sets of strings
    for one set of files.
    """
    return {
        str((root / line).absolute().relative_to(root.absolute()))
        for line in ripgrep_files(root, argument)
    }


@pytest.fixture
def named_ignored_tree(tmp_path) -> Path:
    """Build a repository whose `.gitignore` excludes a directory and a file in it."""
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "igndir/\n*.log\n")
    write(tmp_path / "igndir" / "x.txt")
    write(tmp_path / "igndir" / "y.log")
    write(tmp_path / "igndir" / "sub" / "deep.txt")
    write(tmp_path / "keep.txt")
    return tmp_path


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda _root: "igndir", id="bare"),
        pytest.param(lambda _root: "./igndir", id="dot-relative"),
        pytest.param(lambda root: str(root / "igndir"), id="absolute"),
    ],
)
def test_naming_a_gitignored_directory_searches_it(named_ignored_tree, spelling):
    """Ignore rules govern discovery, not arguments: a named path is searched.

    Every spelling is the same argument, and the rule that excludes it is the same
    rule; toko answered `No files found matching criteria` and exited 1 for all three,
    which is not a missing answer but a wrong one stated out loud.
    """
    argument = spelling(named_ignored_tree)

    found = listed_by_toko(named_ignored_tree, argument)

    assert found == listed_by_ripgrep(named_ignored_tree, argument)
    assert found == {"igndir/sub/deep.txt", "igndir/x.txt"}


def test_a_rule_matching_inside_the_named_directory_still_applies(named_ignored_tree):
    """`*.log` reaches `igndir/y.log` even though `igndir/` no longer reaches it.

    This is the half that separates exempting the argument from switching the ignore
    rules off underneath it: both list `x.txt`, and only the first drops `y.log`.
    """
    found = listed_by_toko(named_ignored_tree, "igndir")

    assert found == listed_by_ripgrep(named_ignored_tree, "igndir")
    assert "igndir/x.txt" in found
    assert "igndir/y.log" not in found


def test_walking_the_parent_still_excludes_the_ignored_directory(named_ignored_tree):
    """Leave the directory nobody named exactly where it was.

    It is discovered, so it is judged, so it is pruned with everything under it.
    """
    found = listed_by_toko(named_ignored_tree, ".")

    assert found == listed_by_ripgrep(named_ignored_tree, ".")
    assert found == {"keep.txt"}


@pytest.mark.parametrize("argument", ["igndir/x.txt", "igndir/y.log", "keep.txt"])
def test_naming_a_file_directly_lists_it_however_it_is_ignored(
    named_ignored_tree, argument
):
    """A named file is an argument too -- under a pruned directory, or excluded itself.

    This already held, and it depends on the same boundary the directory case moves,
    so it is pinned rather than left to be noticed later.
    """
    assert listed_by_toko(named_ignored_tree, argument) == {argument}
    assert listed_by_ripgrep(named_ignored_tree, argument) == {argument}


@pytest.fixture
def nested_ignored_tree(tmp_path) -> Path:
    """Build an ignored directory holding two more, barred from above and from within.

    `blocked/` is named in the root `.gitignore`, the same file that excludes `outer`
    itself, so the exemption has to reach one of that file's rules without reaching
    the other. `own/` comes from `outer`'s own `.gitignore`, which is only read
    because the walk starts here.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "outer/\nouter/blocked/\n")
    write(tmp_path / "outer" / ".gitignore", "own/\n")
    write(tmp_path / "outer" / "top.txt")
    write(tmp_path / "outer" / "blocked" / "a.txt")
    write(tmp_path / "outer" / "own" / "b.txt")
    write(tmp_path / "outer" / "kept" / "c.txt")
    return tmp_path


def test_gitignored_subdirectories_of_the_named_directory_are_still_pruned(
    nested_ignored_tree,
):
    found = listed_by_toko(nested_ignored_tree, "outer")

    assert found == listed_by_ripgrep(nested_ignored_tree, "outer")
    assert found == {"outer/top.txt", "outer/kept/c.txt"}
