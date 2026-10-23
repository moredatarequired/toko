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


def spelled_from(root: Path, cwd: Path, *, absolute: bool = False) -> str:
    """How the walk root is named on a command line run from `cwd`.

    Absolute is a spelling of its own and not a detail of the harness: ripgrep matches
    each path as the caller wrote it, so an absolutely named root strips against the
    working directory nowhere and no anchored rule reaches the walk.
    """
    return str(root.absolute()) if absolute else os.path.relpath(root, cwd)


def run_ripgrep(
    root: Path, *args: str, cwd: Path | None = None, absolute: bool = False
) -> subprocess.CompletedProcess[str]:
    r"""List `root` with ripgrep, from `cwd` when the walk is not started inside it.

    What this helper can express is what the parity suite can cover, and twice now the
    gap has been here rather than in any one test. It used to run ripgrep with its
    working directory always equal to the walk root, which made the shapes that expose
    a rule anchored at the working directory -- core.excludesFile is one -- inexpressible;
    and it spelled the walk root relatively always, which made an absolutely named root
    inexpressible even after `cwd` arrived. Fixtures cannot close a hole in the shapes
    the helper can produce, so widen this before adding one.

    The output is decoded here rather than by `text=True`, which reads the pipe in
    universal-newline mode: that turns the `\r` in a filename into a newline and splits
    one listed path into two, making a name holding a carriage return inexpressible.
    """
    target = [] if cwd is None else ["--", spelled_from(root, cwd, absolute=absolute)]
    result = subprocess.run(  # noqa: S603
        [RIPGREP, "--files", *args, *target],
        cwd=root if cwd is None else cwd,
        capture_output=True,
        check=False,
    )
    decoded = subprocess.CompletedProcess(
        result.args,
        result.returncode,
        os.fsdecode(result.stdout),
        os.fsdecode(result.stderr),
    )
    # 2 is ripgrep's "something went wrong", which --follow returns for this tree's
    # dangling link and its cycle while still listing every file it did reach.
    assert decoded.returncode in (0, 2), decoded.stderr
    return decoded


def ripgrep_files(
    root: Path, *args: str, cwd: Path | None = None, absolute: bool = False
) -> set[str]:
    listed = set(
        run_ripgrep(root, *args, cwd=cwd, absolute=absolute).stdout.split("\n")
    ) - {""}
    if cwd is None:
        return listed
    # ripgrep echoes the path it was given, so its listing carries whatever spelling
    # the root was named with; the comparison is against toko's, which is relative to
    # the walk root.
    prefix = f"{spelled_from(root, cwd, absolute=absolute)}/"
    return {name.removeprefix(prefix) for name in listed}


def toko_files(
    root: Path,
    *,
    hidden: bool,
    follow: bool = False,
    cwd: Path | None = None,
    absolute: bool = False,
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
            Path(spelled_from(root, workdir, absolute=absolute)),
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


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
@pytest.mark.parametrize("workdir", ["", "sub", "outside"])
@pytest.mark.parametrize("walk", ["", "sub"])
def test_the_global_excludes_file_is_anchored_at_the_working_directory(
    anchored_excludes_tree, tmp_path_factory, walk, workdir, spelling
):
    """Every combination of walk start, working directory, and spelling of the root.

    `workdir="sub"` is the one that reads the same excludes file to a different answer
    -- `/x.txt` then names `sub/x.txt` and `sub/y.txt` names nothing -- and `outside`
    is the case where nothing anchored can be resolved at all. git would answer all
    three from the repository root; ripgrep answers them from the working directory,
    and so does toko -- deliberately, and recorded as issue 133.

    `spelling="absolute"` is the only shape here that holds the empty anchor in place:
    an absolutely named root strips against the working directory nowhere, so ripgrep
    drops nothing anchored from it, and anchoring it at the working directory anyway
    loses the `x.txt` ripgrep lists. The relative shapes, `outside` included, all pass
    with the empty anchor removed -- a path that climbs out still carries its `..`
    into the match, where an anchored rule misses it either way.
    """
    root = anchored_excludes_tree / walk if walk else anchored_excludes_tree
    if workdir == "outside":
        cwd = tmp_path_factory.mktemp("elsewhere")
    else:
        cwd = anchored_excludes_tree / workdir if workdir else anchored_excludes_tree
    absolute = spelling == "absolute"

    assert toko_files(root, hidden=False, cwd=cwd, absolute=absolute) == ripgrep_files(
        root, cwd=cwd, absolute=absolute
    )


@pytest.fixture
def segment_excludes_tree(tmp_path, isolated_git_env, monkeypatch) -> Path:
    """Build a tree whose global rule bites one segment below the walk root.

    `*/x.txt` is the shortest rule that can tell a walk root spelled `..` from the
    same directory named any other way: as ripgrep spells that walk, the root's own
    `x.txt` is `../x.txt` and matches, while `sub/x.txt` is `../sub/x.txt` and does
    not.
    """
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    excludes = write(isolated_git_env / "segment-excludes", "*/x.txt\n")
    write(gitconfig, f"[core]\n\texcludesFile = {excludes}\n")

    run_git(tmp_path, "init", "-q")
    for name in ("x.txt", "keep.txt", "sub/x.txt", "sub/keep.txt"):
        write(tmp_path / name)
    return tmp_path


def test_a_walk_root_spelled_through_a_parent_is_matched_through_that_parent(
    segment_excludes_tree,
):
    """A `..` in the walk root stays in every path judged, the way ripgrep keeps it.

    Normalising the root before deciding what it strips against reads `..` as a
    directory outside the working directory, drops the anchor for the whole walk, and
    leaves this rule matching nothing -- while ripgrep, matching the path as spelled,
    drops the file.
    """
    found = toko_files(
        segment_excludes_tree, hidden=False, cwd=segment_excludes_tree / "sub"
    )

    assert found == ripgrep_files(
        segment_excludes_tree, cwd=segment_excludes_tree / "sub"
    )
    # The tree proves something only because the rule has to bite through the `..`.
    assert found == {"keep.txt", "sub/keep.txt", "sub/x.txt"}


@pytest.fixture
def parent_spelled_tree(tmp_path) -> Path:
    """Build a tree where the working directory's own ignore files must not govern.

    `sub` is where both walks are run from, and it holds rules that reach in both
    directions: `*.drop` excludes, `!*.log` re-includes. Neither may reach a walk that
    starts anywhere but inside `sub` -- `sub` is a child of one walk root and a sibling
    of the other, and the ignore files a walk answers to are the ones at or above its
    root. `tmp_path/.ignore` is the genuine ancestor that both walks do answer to.
    """
    write(tmp_path / ".ignore", "*.log\n")
    write(tmp_path / "a" / "sub" / ".ignore", "!*.log\n*.drop\n")
    write(tmp_path / "a" / "sub" / "inner.txt")
    write(tmp_path / "a" / "keep.txt")
    write(tmp_path / "a" / "hidden-by-the-ancestor.log")
    write(tmp_path / "other" / "kept.drop")
    write(tmp_path / "other" / "kept.txt")
    return tmp_path


def test_the_working_directorys_ignore_files_govern_neither_a_parent_nor_a_sibling(
    parent_spelled_tree,
):
    """Both directions of one defect, in one run, because a fix can restore either alone.

    Deriving the ancestors from `Path.absolute()` left `..` in place, so `/w/sub/..`
    reported its parents as `/w/sub` first -- the working directory, a *child* of the
    root being walked. Its ignore files were then installed as parent layers over the
    whole walk, and being the deepest they outranked the real ancestor's.

    Walking `..` is the direction that reaches a provider: `!*.log` overrode the
    ancestor's `*.log` and toko counted, read and sent a file ripgrep excludes. Walking
    `../other` is the quieter one: `*.drop` matched a sibling tree it has no claim on
    and a file ripgrep lists was dropped from the count in silence. Restoring one of
    these while leaving the other is a plausible half-fix, so they are asserted
    together rather than in two tests that could go green apart.
    """
    cwd = parent_spelled_tree / "a" / "sub"
    parent = parent_spelled_tree / "a"
    sibling = parent_spelled_tree / "other"

    up = toko_files(parent, hidden=False, cwd=cwd)
    across = toko_files(sibling, hidden=False, cwd=cwd)

    assert up == ripgrep_files(parent, cwd=cwd)
    assert across == ripgrep_files(sibling, cwd=cwd)
    # The tree proves something only because each rule has to fail to reach.
    assert up == {"keep.txt", "sub/inner.txt"}
    assert across == {"kept.drop", "kept.txt"}


@pytest.fixture
def both_quantities_tree(tmp_path, isolated_git_env, monkeypatch) -> Path:
    """Build one walk that needs the root spelled and the root resolved at once.

    `*/x.txt` in the global excludes file is resolved against the working directory and
    so has to bite *through* the `..`, which only the spelling carries. `*.log` in the
    ancestor above the repository has to beat the `!*.log` in the working directory,
    which only the resolved root gets right. One walk answers to both.
    """
    gitconfig = isolated_git_env / ".gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    excludes = write(isolated_git_env / "segment-excludes", "*/x.txt\n")
    write(gitconfig, f"[core]\n\texcludesFile = {excludes}\n")

    write(tmp_path / ".ignore", "*.log\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    write(repo / "sub" / ".ignore", "!*.log\n")
    for name in ("x.txt", "keep.txt", "drop.log", "sub/x.txt", "sub/keep.txt"):
        write(repo / name)
    return repo


def test_the_root_as_spelled_and_the_root_resolved_are_both_live_in_one_walk(
    both_quantities_tree,
):
    """The two uses of the walk root have opposite needs; neither may serve the other.

    Normalise the root the excludes file is anchored against and `*/x.txt` stops
    reaching `../x.txt`, which ripgrep drops. Leave the root un-normalised where the
    ancestors are read from and the working directory's `!*.log` comes back as a parent
    layer, and `drop.log` is counted, which ripgrep excludes. One value cannot be both,
    so a fix that feeds one quantity to both uses fails here whichever way it leans.
    """
    cwd = both_quantities_tree / "sub"

    found = toko_files(both_quantities_tree, hidden=False, cwd=cwd)

    assert found == ripgrep_files(both_quantities_tree, cwd=cwd)
    # Each name is here because one of the two quantities decides it: x.txt needs the
    # spelling, drop.log needs the resolution, and sub/x.txt holds the spelling honest
    # by being the path the same rule must *not* reach.
    assert found == {"keep.txt", "sub/keep.txt", "sub/x.txt"}


@pytest.fixture
def symlinked_root_tree(tmp_path) -> Path:
    """Build a walk root that is a link into a tree other ignore files govern.

    `A/link` and `B/real` name one directory by two paths, and what sits above each
    of them disagrees: `A/.ignore` drops `*.aa`, `B/.ignore` drops `*.bb`, and the
    repository at `B` drops `*.gg`. ripgrep answers to the directory the link lands
    in, so the walk is held to `B`'s ignore file and to `B`'s repository -- neither of
    which any spelling of the root names. `f.aa` is the file that survives only
    because `A`'s rule does *not* reach, so the two directions are separable.
    """
    write(tmp_path / "A" / ".ignore", "*.aa\n")
    write(tmp_path / "B" / ".ignore", "*.bb\n")
    write(tmp_path / "B" / ".gitignore", "*.gg\n")
    run_git(tmp_path / "B", "init", "-q")
    for name in ("f.aa", "f.bb", "f.gg", "keep.txt"):
        write(tmp_path / "B" / "real" / name)
    (tmp_path / "A" / "link").symlink_to(tmp_path / "B" / "real")
    return tmp_path


def test_a_symlinked_walk_root_answers_to_the_ignore_files_over_its_target(
    symlinked_root_tree,
):
    """`Path.resolve`, not `os.path.abspath`: a link lands where it points.

    Both spell `A/link` absolutely and only `resolve` follows it, so `abspath` reads
    the ancestors as `A` and drops `f.aa` while keeping the `f.bb` that `B`'s own
    ignore file excludes -- an answer ripgrep never gives, and the exact substitution
    a later simplification of this line would make.
    """
    root = symlinked_root_tree / "A" / "link"
    cwd = symlinked_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert found == ripgrep_files(root, cwd=cwd)
    # `f.aa` survives because `A`'s rule must not reach and `f.bb` is gone because
    # `B`'s must: reading the ancestors from the unresolved root swaps the pair.
    assert found == {"f.aa", "keep.txt"}


def test_the_repository_over_a_symlinked_walk_root_is_found_through_the_link(
    symlinked_root_tree,
):
    """The repository a walk sits in is looked for from the root resolved.

    Asking `_find_repo_root` about the root as named searches `A/link`, `A` and the
    directories above them, finds no `.git`, and takes the walk for one outside any
    repository -- which switches off every git ignore file, `B/.gitignore` with them,
    and leaks a file ripgrep excludes. The listing above cannot be split from this on
    its own, so `f.gg` is asserted by name.
    """
    root = symlinked_root_tree / "A" / "link"
    cwd = symlinked_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert "f.gg" not in found
    assert found == ripgrep_files(root, cwd=cwd)


@pytest.fixture
def anchored_ancestor_tree(tmp_path) -> Path:
    """Build an anchored ignore rule over a walk root spelled through a parent.

    `/keep.txt` is anchored at the directory holding the ignore file, so it reaches
    that directory's own `keep.txt` and leaves `sub/keep.txt` alone. Walking from
    `sub` names the root `..`, which every path judged then carries.
    """
    write(tmp_path / ".ignore", "/keep.txt\n")
    for name in ("keep.txt", "other.txt", "sub/keep.txt", "sub/other.txt"):
        write(tmp_path / name)
    return tmp_path


def test_an_anchored_rule_over_a_root_spelled_through_a_parent_still_reaches_it(
    anchored_ancestor_tree,
):
    """The layers built over the walk root strip the root as spelled, not resolved.

    `keep.txt` is judged as `/w/sub/../keep.txt`, so an anchor at the resolved root
    strips `/w/` and leaves `sub/../keep.txt` for `/keep.txt` to miss. Nothing
    complains: `str.removeprefix` hands back a prefix it did not find, so the wrong
    anchor reads as a path that simply does not match, and the file ripgrep drops is
    counted, read and sent instead.
    """
    cwd = anchored_ancestor_tree / "sub"

    found = toko_files(anchored_ancestor_tree, hidden=False, cwd=cwd)

    assert found == ripgrep_files(anchored_ancestor_tree, cwd=cwd)
    # `sub/keep.txt` is the path the same rule must not reach, which is what keeps
    # the anchor honest rather than merely present.
    assert found == {"other.txt", "sub/keep.txt", "sub/other.txt"}


@pytest.fixture
def rule_above_the_root_tree(tmp_path) -> Path:
    """Build a tree whose rule sits one directory above the walk root.

    Every walk here starts at `root` and the rule under test is written into
    `tmp_path/.gitignore`, so it is read as an ancestor's rather than as the walk
    root's own -- the two are matched differently, and the same rule gives opposite
    answers from the two places. `a.bb` and `sub/deep.bb` share a suffix at two
    different depths and `keep.txt` and `sub/keep.txt` share a name, so a rule can
    reach one depth, both, or neither, and the listing says which.

    The repository is at `tmp_path` and not at `root`, because a `.gitignore` above
    the walk root is only read at all while the walk is inside the repository holding
    it -- which is also the condition ripgrep puts on reading it.
    """
    run_git(tmp_path, "init", "-q")
    for name in ("a.bb", "keep.txt", "sub/deep.bb", "sub/keep.txt"):
        write(tmp_path / "root" / name)
    return tmp_path


@pytest.mark.parametrize(
    ("rule", "kept"),
    [
        # The rule that names the file's real path, and reaches nothing: the entry is
        # put to it as `root/deep.bb`, which has no `sub` in it to match.
        ("/root/sub/deep.bb", {"a.bb", "keep.txt", "sub/deep.bb", "sub/keep.txt"}),
        # The same rule one segment shorter, reaching both depths at once. `*` does
        # not cross a separator, but there is no separator left to cross once the
        # directory between the walk root and the entry has been dropped.
        ("/root/*.bb", {"keep.txt", "sub/keep.txt"}),
        # Spelled without the leading slash, which changes nothing: a rule carrying a
        # separator is anchored either way.
        ("root/*.bb", {"keep.txt", "sub/keep.txt"}),
        # A name rather than a glob, so the reach cannot be read as loose matching:
        # one rule takes both `keep.txt` files, because both are put to it as
        # `root/keep.txt`.
        ("/root/keep.txt", {"a.bb", "sub/deep.bb"}),
        # And a rule naming the intervening directory reaches nothing at all, which is
        # what stops the whole shape reading as "an anchored rule up there is skipped".
        ("/root/sub/*.bb", {"a.bb", "keep.txt", "sub/deep.bb", "sub/keep.txt"}),
    ],
)
def test_a_rule_above_the_walk_root_reaches_what_ripgrep_has_it_reaching(
    rule_above_the_root_tree, rule, kept
):
    """An ignore file above the walk root is matched against names, not against paths.

    ripgrep joins the walk root onto each entry's own name for these layers and drops
    whatever lies between, so `root/sub/deep.bb` is judged as `root/deep.bb`. Spelling
    them the way git does -- the entry's whole path below the root -- misses in both
    directions at once, and the two rules at the top of this table are one file apiece:
    `/root/sub/deep.bb` then drops a file ripgrep lists, and `/root/*.bb` then keeps a
    file ripgrep drops. They are parametrized rather than asserted together because the
    single spelling decides both, so neither can go green without the other.
    """
    write(rule_above_the_root_tree / ".gitignore", f"{rule}\n")
    root = rule_above_the_root_tree / "root"
    cwd = rule_above_the_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert found == ripgrep_files(root, cwd=cwd)
    assert found == kept


def test_the_walk_roots_own_ignore_file_still_matches_whole_paths(
    rule_above_the_root_tree,
):
    """The other half of the asymmetry: only a file *above* the root is name-matched.

    `/*.bb` in the walk root's own `.gitignore` is the counterpart of `/root/*.bb` in
    the parent's, and ripgrep answers the two differently -- it takes `a.bb` alone here
    and both `.bb` files there. Matching every layer against names would take both
    here too, and there is nothing else in the suite that would notice.
    """
    write(rule_above_the_root_tree / "root" / ".gitignore", "/*.bb\n")
    root = rule_above_the_root_tree / "root"
    cwd = rule_above_the_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert found == ripgrep_files(root, cwd=cwd)
    assert found == {"keep.txt", "sub/deep.bb", "sub/keep.txt"}


def test_a_negation_above_the_walk_root_follows_the_same_spelling(
    rule_above_the_root_tree,
):
    """The re-include is matched the same way the exclusion was, or it cannot bite.

    `/root/**/*.bb` reaches both depths whichever spelling is used, so the exclusion
    is not what this turns on: `!/root/deep.bb` is, and it names no path in the tree,
    since `deep.bb` lives in `sub`. ripgrep re-includes the file anyway, because a
    name is exactly what the rules are put the file under. Match these layers against
    whole paths and the negation finds nothing to re-include, and the file goes.
    """
    write(rule_above_the_root_tree / ".gitignore", "/root/**/*.bb\n!/root/deep.bb\n")
    root = rule_above_the_root_tree / "root"
    cwd = rule_above_the_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert found == ripgrep_files(root, cwd=cwd)
    assert found == {"keep.txt", "sub/deep.bb", "sub/keep.txt"}


def test_a_directory_rule_above_the_walk_root_prunes_by_name(rule_above_the_root_tree):
    """A directory is put to those rules under its name too, and pruned on it.

    `sub/nested` is two directories below the walk root and `/root/nested/` names it
    at one, so the rule reaches it only if the directory between them is dropped the
    way a file's is -- and ripgrep prunes it. The trailing separator is the other half:
    a spelling that took the name of `root/sub/nested/` rather than of `root/sub/nested`
    would come away with an empty string and prune nothing at all.
    """
    write(rule_above_the_root_tree / "root" / "sub" / "nested" / "inner.txt")
    write(rule_above_the_root_tree / ".gitignore", "/root/nested/\n")
    root = rule_above_the_root_tree / "root"
    cwd = rule_above_the_root_tree

    found = toko_files(root, hidden=False, cwd=cwd)

    assert found == ripgrep_files(root, cwd=cwd)
    assert found == {"a.bb", "keep.txt", "sub/deep.bb", "sub/keep.txt"}


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


# The rule vocabulary these tests sweep. `pathspec` strips both ends of a rule; git and
# ripgrep strip only the trailing end, so every shape below is one the previous
# vocabulary -- trailing whitespace only -- had no way to write, which is why a
# both-directions divergence sat in the ignore-rule pipeline unmeasured.
LEADING_WHITESPACE_RULES = [
    "  x.txt",
    "\tx.txt",
    " /x.txt",
    "  *",
    "  *.log",
    "  build/",
    "  # note",
    "  !x.txt",
    "  x.txt  ",
    "  x.txt\\ ",
    "   ",
    "",
]

# Each rule shape has to meet both spellings of every name it could be read as, or a
# rule matching the wrong file looks the same as a rule matching nothing.
WHITESPACE_NAMES = [
    "keep.txt",
    "x.txt",
    "  x.txt",
    "\tx.txt",
    "  x.txt ",
    "a.log",
    "  a.log",
    "  # note",
    "  !x.txt",
    "build/f.txt",
    "  build/f.txt",
    "sub/x.txt",
    "sub/  x.txt",
]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("rule", LEADING_WHITESPACE_RULES)
def test_a_rules_leading_whitespace_is_part_of_the_name_it_writes(
    tmp_path, rule, newline
):
    r"""Leading whitespace is literal for git and ripgrep; only the trailing end strips.

    A reader that strips both ends diverges twice from one line: the file the rule
    actually names is counted, and its contents go to whatever provider is counting,
    while a file sharing the stripped name is dropped from the count instead. Neither
    half is visible in an editor. CRLF is swept alongside because a `\r` reaches the
    strip as trailing whitespace and could plausibly be what removes the leading run.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(f"{rule}{newline}".encode())
    for name in WHITESPACE_NAMES:
        write(tmp_path / name)

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)


def test_a_leading_space_drops_the_name_it_writes_and_spares_the_bare_one(tmp_path):
    """The two directions of the divergence, pinned as an exact set rather than parity.

    The parity sweep above passes if toko and ripgrep are wrong together, which is the
    one way a differential test can go quiet. This says which files survive.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "  x.txt\n")
    write(tmp_path / "x.txt")
    write(tmp_path / "  x.txt")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt", "x.txt"}


def test_an_indented_star_is_not_a_rule_that_ignores_the_whole_tree(tmp_path):
    """The severe shape, and an ordinary one to write: somebody indents a block.

    Stripped to `*` the rule matches every file, so the walk finds nothing at all and
    the run reports no files rather than a wrong number.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "  *\n")
    write(tmp_path / "README.md")
    write(tmp_path / "src" / "a.py")
    write(tmp_path / "src" / "b.py")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"README.md", "src/a.py", "src/b.py"}


def test_a_whitespace_only_rule_stays_the_no_operation_git_reads_it_as(tmp_path):
    """Trailing whitespace still strips, and a rule that is only whitespace is blank.

    The escape that holds leading whitespace cannot be applied here: git reads the line
    as blank once its trailing whitespace is gone, and a rule that is a lone backslash
    is one `pathspec` rejects outright.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"   \n\t\n")
    write(tmp_path / "keep.txt")
    write(tmp_path / "   ")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt", "   "}


# Ignore files whose bytes are not valid UTF-8. ripgrep decodes a line at a time and
# stops at the first line it cannot read, keeping the rules above it: it neither skips
# the one bad line nor discards the file. Another shape the vocabulary could not write.
ESCAPED_AND_EXOTIC_RULES = [
    "\\ x.txt",  # the user's own way to write a literal leading space
    "\\ \\ x.txt",
    "\\\tx.txt",  # backslash before a real tab
    "!  x.txt",
    "\\!x.txt",
    "X.TXT",
    "x\u00e9.txt",  # NFC: e-acute as one code point
    "xe\u0301.txt",  # NFD: e plus a combining acute
]

EXOTIC_NAMES = [
    "keep.txt",
    "x.txt",
    " x.txt",
    "  x.txt",
    "\tx.txt",
    "X.TXT",
    "!x.txt",
    "!",
    "x\u00e9.txt",
    "xe\u0301.txt",
]


@pytest.mark.parametrize("rule", ESCAPED_AND_EXOTIC_RULES)
def test_a_rule_the_user_escaped_still_names_what_ripgrep_reads_it_as(tmp_path, rule):
    """Neither ripgrep nor git folds case or Unicode normalisation, and nor may toko.

    One accented name written NFC and the same name written NFD are two separate
    files on disk, and a rule matches whichever one shares its bytes. A reader that
    normalised either side would match both -- dropping a file ripgrep lists -- or
    neither.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(f"{rule}\n".encode())
    for name in EXOTIC_NAMES:
        write(tmp_path / name)

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)


def test_a_negation_keeps_the_leading_whitespace_that_follows_its_bang(tmp_path):
    """`!  x.txt` takes back the file named with two spaces, not the bare one.

    Negation is the leaking direction. Read as `!x.txt` it returns a file the user
    excluded with `*` to the count and sends its contents to whatever is counting,
    while the file the line actually names stays dropped -- both errors from one rule.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ".gitignore", "*\n!  x.txt\n")
    write(tmp_path / "x.txt")
    write(tmp_path / "  x.txt")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"  x.txt"}


def test_a_byte_order_mark_belongs_to_the_first_rule_the_way_ripgrep_reads_it(tmp_path):
    """A third deliberate divergence from `git check-ignore`, measured not assumed.

    git strips a UTF-8 BOM off an ignore file's first line and applies the rule beneath
    it; ripgrep leaves the BOM in the pattern, where it matches no filename at all. In
    this tree `git check-ignore` calls `x.txt` ignored and `rg --files` lists it, and
    ripgrep is what this walk is measured against.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"\xef\xbb\xbfx.txt\n")
    write(tmp_path / "x.txt")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt", "x.txt"}


# Every ignore source is parsed by the same grammar, so a rule shape measured on one
# file proves nothing about the others. This is the sweep that catches a fix applied
# where the fixtures happen to look, rather than where the parsing actually happens.
IGNORE_FILE_KINDS = [".gitignore", ".ignore", ".rgignore", ".git/info/exclude"]


@pytest.mark.parametrize("ignore_file", IGNORE_FILE_KINDS)
def test_leading_whitespace_is_literal_in_every_kind_of_ignore_file(
    tmp_path, ignore_file
):
    """Every ignore source parses rules alike, so every one of them has to be measured.

    `.ignore` and `.rgignore` apply with no repository at all, so getting them wrong
    reaches further than getting `.gitignore` wrong does, not less far.
    """
    run_git(tmp_path, "init", "-q")
    write(tmp_path / ignore_file, "  x.txt\n")
    write(tmp_path / "x.txt")
    write(tmp_path / "  x.txt")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt", "x.txt"}


UNDECODABLE_IGNORE_FILES = [
    b"x\xe9.txt\n*.log\n",
    b"*.log\nx\xe9.txt\n",
    b"*.log\nx\xe9.txt\n*.md\n",
    b"*.log\nx\xe9.txt\n!y.log\n",
    b"\xff\xfe\x00\x01\n",
    b"\xe9*.log\n*.md\n",
    b"*.log\r\nx\xe9.txt\r\n*.md\r\n",
    b"*.log\nx\xe9.txt",
    b"*.log\nx\xe9y\n!y.log\n",
]


@pytest.mark.parametrize("raw", UNDECODABLE_IGNORE_FILES, ids=repr)
def test_an_undecodable_line_truncates_an_ignore_file_where_ripgrep_truncates_it(
    tmp_path, raw
):
    """Where the bad line sits decides the answer, so the sweep varies where it sits.

    A reader that replaces the undecodable bytes and carries on keeps applying every
    later rule; one that discards the file drops the earlier ones. The two are told
    apart only by a file with rules on both sides of the bad line.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(raw)
    write(tmp_path / "keep.md")
    write(tmp_path / "y.log")
    write(tmp_path / "z.txt")

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)


def test_a_negation_below_an_undecodable_line_never_re_includes_the_file(tmp_path):
    """The leaking direction: reading past the bad line re-includes what rg never lists.

    `*.log` excludes the file and `!y.log` would take it back, but ripgrep never reaches
    the negation. A reader that does counts `y.log` and sends its contents to whichever
    provider is counting -- an exclusion the user wrote, undone by a byte.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\nx\xe9.txt\n!y.log\n")
    write(tmp_path / "y.log")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt"}


def test_an_undecodable_first_line_leaves_every_later_rule_unread(tmp_path):
    """Truncation, told apart from skipping just the bad line, at the sharpest position.

    With the failure on line 1 there is nothing above it, so a truncating reader applies
    no rule at all while a skipping reader still applies `*.md`.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"\xe9*.log\n*.md\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "y.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "y.log"}


@pytest.mark.parametrize("ignore_file", IGNORE_FILE_KINDS)
def test_an_undecodable_line_truncates_every_kind_of_ignore_file(tmp_path, ignore_file):
    """The truncation is a property of reading an ignore file, not of one filename."""
    run_git(tmp_path, "init", "-q")
    path = tmp_path / ignore_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"*.log\nx\xe9.txt\n!y.log\n")
    write(tmp_path / "y.log")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt"}


# A lone carriage return is not a line ending to ripgrep. `BufRead::lines` ends a line
# at a newline and nowhere else, taking the `\r` of a CRLF with it, so a `\r` on its own
# is a byte inside the rule. The CRLF sweep above cannot see this: every `\r` in it is
# followed by a newline, which is the one arrangement both readings agree on.
LONE_CARRIAGE_RETURN_IGNORE_FILES = [
    b"*.log\r*.md\r",
    b"*.log\r*.md",
    b"*.log\r",
    b"*.log\r\r\n",
    b"x\r.txt\n",
    b"*.log\r\n*.md\rx.txt\n",
    b"*.log\n*.md\r!y.log\n",
    b" \ry.log\n",
    b"\r y.log\n",
    b"*.log\\ \r\n",
    b"\\\\ \r\r\n",
    b"*.log\rcaf\xe9.txt\r!y.log\r",
    b"*.log\r\xff*.md\n",
    b"*.log\xff\r*.md\n",
]

CARRIAGE_RETURN_NAMES = [
    "keep.md",
    "x.txt",
    "y.log",
    "x\r.txt",
    " y.log",
    " \ry.log",
    "\r y.log",
    "y.log ",
    "\\",
    "\\ ",
]


@pytest.mark.parametrize("raw", LONE_CARRIAGE_RETURN_IGNORE_FILES, ids=repr)
def test_a_lone_carriage_return_does_not_end_a_rule_for_ripgrep(tmp_path, raw):
    r"""A reader that breaks on `\r` applies as two rules a line ripgrep never split.

    Both halves are wrong at once and neither is visible in an editor: the text before
    the `\r` becomes a rule ripgrep never had, and the whole line stops naming the file
    it does name. The sweep varies where the `\r` sits, because the halves it produces
    are what decide the answer.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(raw)
    for name in CARRIAGE_RETURN_NAMES:
        write(tmp_path / name)

    assert toko_files(tmp_path, hidden=False) == ripgrep_files(tmp_path)


def test_a_carriage_return_separated_file_holds_no_rule_at_all(tmp_path):
    """The severe shape: an editor writes CR line endings and every rule disappears.

    Read as three rules the file excludes two of these; read the way ripgrep reads it
    the whole file is one glob that names nothing. Saying which files survive keeps the
    sweep above from passing because toko and ripgrep are wrong together.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\r*.md\r")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "y.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt", "y.log"}


def test_a_carriage_return_inside_a_rule_still_names_the_file_holding_one(tmp_path):
    r"""The `\r` belongs to the name, so the rule matches the file whose name has it.

    A splitting reader spares `x\r.txt` -- the file the line actually names -- and
    counts it, sending its contents to whatever provider is counting, while `x.txt`
    survives either way and hides the difference from a test that only counts files.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"x\r.txt\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "x\r.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt"}


def test_a_carriage_return_beside_a_crlf_line_ending_leaves_the_crlf_working(tmp_path):
    r"""CRLF still ends a line, told apart from the lone `\r` inside the same file.

    A reader that stops taking the `\r` off a CRLF line keeps it in the rule, where it
    matches nothing; one that goes on breaking at every `\r` splits the second line.
    Only a file holding both arrangements separates the two.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\r\n*.md\rx.txt\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "y.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt"}


def test_a_negation_after_a_carriage_return_never_re_includes_the_file(tmp_path):
    r"""The leaking direction: `*.md\r!y.log` is one glob, not an exclusion undone.

    A splitting reader reads a `!y.log` ripgrep never sees and hands back a file the
    line above excluded, so its contents reach whichever provider is counting -- and
    it drops `keep.md` from the same line, which is the error that gets noticed.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\n*.md\r!y.log\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "y.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt"}


def test_a_carriage_return_before_an_undecodable_byte_truncates_the_whole_line(
    tmp_path,
):
    r"""Where the line ends decides what the undecodable byte takes with it.

    `*.log\r\xff*.md` is a single line, so the byte ripgrep cannot read costs the whole
    file its rules -- `*.log` included. A reader that breaks at the `\r` first hands
    `*.log` to the spec before the truncation can reach it and drops a file ripgrep
    lists, which is why the strip belongs after the decode rather than before the split.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\r\xff*.md\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "y.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt", "y.log"}


def test_a_crlf_line_ending_comes_off_a_rule_that_escapes_its_trailing_space(tmp_path):
    r"""The `\r` has to go before the rule is read, not be left for the strip to take.

    `*.log\\ ` keeps its trailing space only because the rule ends with the escape; a
    `\r` left behind it hides the escape, and what remains once the whitespace strips
    is a dangling backslash rather than a pattern. Written with LF the same rule is
    unremarkable, so only the CRLF spelling shows whether the `\r` was taken off.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b"*.log\\ \r\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "x.txt")
    write(tmp_path / "y.log")
    write(tmp_path / "y.log ")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "x.txt", "y.log"}


@pytest.mark.parametrize(
    ("raw", "listed_beside_keep"),
    [(b"\\\\ \n", "\\"), (b"\\\\ \r\r\n", "\\ ")],
    ids=repr,
)
def test_a_carriage_return_behind_an_escape_decides_which_file_a_rule_names(
    tmp_path, raw, listed_beside_keep
):
    r"""One `\r` is the line ending; a second one is trailing whitespace ahead of it.

    `\\ ` keeps its space only because the rule ends with the escape. Leave a `\r`
    behind it and the escape is no longer at the end, so the whitespace strips and the
    rule names the bare backslash instead -- ripgrep reads the two spellings as two
    different files. Taking every trailing `\r` off restores an escape ripgrep had
    already lost and excludes the other file of the pair, which is a one-byte edit to
    the reader and a complete swap of which file's contents a provider is shown.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(raw)
    write(tmp_path / "keep.md")
    write(tmp_path / "\\")
    write(tmp_path / "\\ ")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", listed_beside_keep}


def test_a_carriage_return_beside_leading_whitespace_stays_in_the_name(tmp_path):
    r"""The `\r` and the escape that holds leading whitespace meet on one line.

    ` \ry.log` names a file starting with a space, and the escape has to survive the
    `\r` sitting behind it. A splitting reader turns the line into a blank one and a
    bare `y.log`, which excludes the file ripgrep lists and lists the one it excludes.
    """
    run_git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(b" \ry.log\n")
    write(tmp_path / "keep.md")
    write(tmp_path / "y.log")
    write(tmp_path / " y.log")
    write(tmp_path / " \ry.log")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.md", "y.log", " y.log"}


def test_a_gitignore_directly_above_the_repository_root_reaches_nothing_inside_it(
    tmp_path,
):
    """Where the ancestor search stops, measured one level out rather than at infinity.

    Ancestor `.gitignore` layers apply only within the repository the walk root belongs
    to. The realistic wrong version of that is not a missing boundary but one sitting a
    level too high, and a fixture whose ignore file is far above the root cannot tell
    the two apart -- both wrong versions pass it. This puts the file exactly one level
    above the boundary, where an off-by-one is the whole difference. The rule governs
    whether a file's contents leave the machine, so it is worth more than one assertion.
    """
    outside = tmp_path / "outside"
    repo = outside / "repo"
    repo.mkdir(parents=True)
    run_git(repo, "init", "-q")
    write(outside / ".gitignore", "secret.txt\n")
    write(repo / "secret.txt")
    write(repo / "keep.txt")

    listed = ripgrep_files(repo)

    assert toko_files(repo, hidden=False) == listed
    assert listed == {"keep.txt", "secret.txt"}


def test_a_gitignore_with_no_repository_anywhere_reaches_nothing(tmp_path):
    """The other angle on the same boundary: no repository, so no git layer at all.

    A boundary that is merely too generous still refuses this tree, because there is no
    repository root to be generous about. Only a boundary that is gone entirely reads
    the file -- which is the divergence that hands an excluded file to a provider on a
    directory that was never a checkout.
    """
    write(tmp_path / ".gitignore", "secret.txt\n")
    write(tmp_path / "secret.txt")
    write(tmp_path / "keep.txt")

    listed = ripgrep_files(tmp_path)

    assert toko_files(tmp_path, hidden=False) == listed
    assert listed == {"keep.txt", "secret.txt"}


# Shapes the rule vocabulary had no way to write before, each measured against ripgrep
# rather than reasoned about. The escaped forms matter because the fix now spells rules
# with a backslash of its own: an escape the *user* wrote has to go on meaning what it
# meant. Spelled with explicit escapes because several differ only in bytes on screen.
