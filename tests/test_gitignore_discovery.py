"""Ignore-file discovery: upward search, per-directory stack, ripgrep parity."""

import inspect
import os
import socket
import subprocess
from pathlib import Path

import pytest

from tests.conftest import GIT_LOCATION_VARS
from tests.git_runner import fixture_git_env, run_git
from tests.test_ripgrep_parity import toko_scanned
from toko.file_reader import find_files


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "fx"
    root.mkdir()
    run_git(root, "init", "-q")
    return root


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def names(files: list[Path], base: Path) -> set[str]:
    return {str(f.absolute().relative_to(base)) for f in files}


def test_nested_gitignore_applies_to_its_own_subtree(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / ".gitignore", "node_modules/\ndist\n")
    write(repo / "loadout" / "app.js")
    write(repo / "loadout" / "node_modules" / "pkg" / "index.js")
    write(repo / "loadout" / "dist" / "bundle.js")
    write(repo / "top.log")
    write(repo / "README.md")

    assert names(find_files(repo), repo) == {"README.md", "loadout/app.js"}


def test_nested_rules_do_not_leak_upward(repo):
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "node_modules" / "index.js")
    write(repo / "loadout" / "node_modules" / "index.js")

    assert names(find_files(repo), repo) == {"node_modules/index.js"}


def test_invocation_from_a_subdirectory_finds_repo_root_rules(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / "app.js")
    write(repo / "loadout" / "debug.log")

    found = find_files(repo / "loadout")
    assert names(found, repo) == {"loadout/app.js"}


def test_invocation_on_a_relative_subpath_finds_repo_root_rules(repo, monkeypatch):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / "deep" / "app.js")
    write(repo / "loadout" / "deep" / "debug.log")

    monkeypatch.chdir(repo / "loadout")
    found = find_files(Path("deep"))
    assert found == [Path("deep/app.js")]


def test_shallow_listing_also_uses_repo_root_rules(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / "app.js")
    write(repo / "loadout" / "debug.log")

    found = find_files(repo / "loadout", recursive=False)
    assert names(found, repo) == {"loadout/app.js"}


def test_info_exclude_is_honored(repo):
    write(repo / ".git" / "info" / "exclude", "secret/\n*.bak\n")
    write(repo / "secret" / "key.txt")
    write(repo / "notes.txt.bak")
    write(repo / "notes.txt")

    assert names(find_files(repo), repo) == {"notes.txt"}


def test_core_excludes_file_is_honored(repo, tmp_path):
    excludes = write(tmp_path / "my-excludes", "*.swp\n")
    run_git(repo, "config", "--global", "core.excludesFile", str(excludes))
    write(repo / "notes.txt")
    write(repo / "notes.swp")

    assert names(find_files(repo), repo) == {"notes.txt"}


def test_a_repository_local_core_excludes_file_is_passed_over(repo, tmp_path):
    """Ripgrep parses only git's global config, so a repository's own is passed over.

    tests/test_ripgrep_parity.py holds the differential that establishes this.
    """
    globally = write(tmp_path / "global-excludes", "*.swp\n")
    locally = write(tmp_path / "repo-excludes", "*.tmp\n")
    run_git(repo, "config", "--global", "core.excludesFile", str(globally))
    run_git(repo, "config", "--local", "core.excludesFile", str(locally))
    write(repo / "notes.txt")
    write(repo / "notes.swp")
    write(repo / "notes.tmp")

    assert names(find_files(repo), repo) == {"notes.txt", "notes.tmp"}


def test_xdg_git_ignore_is_the_default_excludes_file(repo, isolated_git_env):
    write(isolated_git_env / ".config" / "git" / "ignore", "*.swp\n")
    write(repo / "notes.txt")
    write(repo / "notes.swp")

    assert names(find_files(repo), repo) == {"notes.txt"}


def test_a_deeper_gitignore_can_negate_a_shallower_one(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "keep" / ".gitignore", "!important.log\n")
    write(repo / "keep" / "important.log")
    write(repo / "keep" / "noisy.log")
    write(repo / "top.log")

    assert names(find_files(repo), repo) == {"keep/important.log"}


def test_a_deeper_gitignore_can_negate_info_exclude(repo):
    write(repo / ".git" / "info" / "exclude", "*.log\n")
    write(repo / "keep" / ".gitignore", "!kept.log\n")
    write(repo / "keep" / "kept.log")
    write(repo / "dropped.log")

    assert names(find_files(repo), repo) == {"keep/kept.log"}


def test_the_gitignore_beside_info_exclude_can_negate_it_too(repo):
    """The same-directory case, where depth cannot decide it and kind order alone does.

    The test above puts the .gitignore deeper, so it wins whichever way the two are
    ranked. Here they sit in one directory, which is the only arrangement that pins
    .gitignore above .git/info/exclude -- as git and `rg --files` both rank them.
    """
    write(repo / ".git" / "info" / "exclude", "*.log\n")
    write(repo / ".gitignore", "!kept.log\n")
    write(repo / "kept.log")
    write(repo / "dropped.log")

    assert names(find_files(repo), repo) == {"kept.log"}


def test_anchored_patterns_are_relative_to_their_own_gitignore(repo):
    write(repo / ".gitignore", "/build\n")
    write(repo / "build" / "out.o")
    write(repo / "pkg" / "build" / "out.o")
    write(repo / "pkg" / ".gitignore", "/target\n")
    write(repo / "pkg" / "target" / "out.o")
    write(repo / "pkg" / "sub" / "target" / "out.o")

    assert names(find_files(repo), repo) == {"pkg/build/out.o", "pkg/sub/target/out.o"}


def test_directory_only_pattern_leaves_a_like_named_file_alone(repo):
    write(repo / "pkg" / ".gitignore", "build/\n")
    write(repo / "pkg" / "build" / "out.o")
    write(repo / "pkg" / "build.txt")
    write(repo / "pkg" / "sub" / "build")

    assert names(find_files(repo), repo) == {"pkg/build.txt", "pkg/sub/build"}


def test_a_nested_repository_contributes_its_own_rules(repo):
    nested = repo / "vendor" / "lib"
    nested.mkdir(parents=True)
    run_git(nested, "init", "-q")
    write(nested / ".gitignore", "target/\n")
    write(nested / "src.rs")
    write(nested / "target" / "build.rs")

    assert names(find_files(repo), repo) == {"vendor/lib/src.rs"}


def test_ignored_directories_are_pruned_rather_than_walked(repo):
    """A gitignored directory is never opened, not merely filtered out afterwards.

    Spied on `os.scandir`, which is the primitive the walk actually uses. The previous
    version of this test patched `os.walk`, which `file_reader` has never called, so
    it recorded an empty list and its "nothing ignored was visited" assertion was true
    of nothing. The assertion that the spy saw the root is what keeps a future rename
    of the walk primitive failing loudly instead of going quiet again.
    """
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "loadout" / "app.js")
    for i in range(5):
        write(repo / "loadout" / "node_modules" / f"pkg{i}" / "index.js")

    scanned = toko_scanned(repo)

    assert "." in scanned, "the spy recorded nothing, so it cannot prove a pruning"
    assert scanned == {".", "loadout"}
    assert names(find_files(repo), repo) == {"loadout/app.js"}


def test_no_ignore_keeps_every_ignored_file(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "loadout" / "node_modules" / "index.js")
    write(repo / "top.log")
    write(repo / "README.md")

    # The positive control belongs in this call, not a sibling test: without it the
    # negative one is equally satisfied by a walk that never reads an ignore file.
    assert names(find_files(repo), repo) == {"README.md"}
    assert names(find_files(repo, respect_gitignore=False), repo) == {
        "README.md",
        "top.log",
        "loadout/node_modules/index.js",
    }


def test_exclude_patterns_compose_with_ignore_discovery(repo):
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "loadout" / "node_modules" / "index.js")
    write(repo / "loadout" / "app.js")
    write(repo / "loadout" / "app.ts")

    found = find_files(repo, exclude_patterns=["*.ts"])
    assert names(found, repo) == {"loadout/app.js"}

    found = find_files(repo, respect_gitignore=False, exclude_patterns=["*.ts"])
    assert names(found, repo) == {"loadout/app.js", "loadout/node_modules/index.js"}


def test_ignore_files_still_apply_when_git_is_not_installed(repo, monkeypatch):
    monkeypatch.setenv("PATH", "")
    write(repo / ".gitignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}


def test_ignore_files_still_apply_when_git_config_fails(repo, tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    failing_git = fake_bin / "git"
    failing_git.write_text("#!/bin/sh\nexit 1\n")
    failing_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    write(repo / ".gitignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}


def test_dot_ignore_applies_like_a_gitignore(repo):
    write(repo / ".ignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}


def test_rgignore_applies_like_a_gitignore(repo):
    write(repo / ".rgignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}


def test_dot_ignore_overrides_gitignore_in_both_directions(repo):
    """A .ignore drops what the .gitignore beside it kept, and keeps what it dropped.

    `strict/.gitignore` says keep and nothing else, so dropping `strict/keep.log` can
    only be `.ignore` outranking it. Spelling that .gitignore as a negation followed
    by `*.log` instead made the git side drop the file on its own, and the strict half
    of "both directions" decided nothing.
    """
    write(repo / "strict" / ".gitignore", "!keep.log\n")
    write(repo / "strict" / ".ignore", "*.log\n")
    write(repo / "strict" / "keep.log")
    write(repo / "loose" / ".gitignore", "*.log\n")
    write(repo / "loose" / ".ignore", "!keep.log\n")
    write(repo / "loose" / "keep.log")

    assert names(find_files(repo), repo) == {"loose/keep.log"}


@pytest.mark.parametrize("dot_file", [".ignore", ".rgignore"])
def test_a_shallow_dot_ignore_outranks_a_deeper_gitignore(repo, dot_file):
    """Depth decides between two files of one kind; kind decides between the kinds.

    Resolved by depth alone a nested .gitignore is the last word, which is what
    ripgrep does *not* do: it ranks every dot-ignore file above every git one first.
    """
    write(repo / dot_file, "keep.txt\n")
    write(repo / "sub" / ".gitignore", "!keep.txt\n")
    write(repo / "sub" / "keep.txt")
    write(repo / "sub" / "other.txt")

    assert names(find_files(repo), repo) == {"sub/other.txt"}


def test_a_deeper_dot_ignore_still_beats_a_shallower_one(repo):
    write(repo / ".ignore", "*.log\n")
    write(repo / "sub" / ".ignore", "!keep.log\n")
    write(repo / "sub" / "keep.log")
    write(repo / "drop.log")

    assert names(find_files(repo), repo) == {"sub/keep.log"}


def test_dot_ignore_above_the_repository_root_still_applies(repo, tmp_path):
    write(tmp_path / ".ignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}


def test_gitignore_above_the_repository_root_does_not_apply(repo, tmp_path):
    # `scratch.tmp` is the live rule: it proves this call reads .gitignore files at
    # all, so `debug.log` surviving is the boundary and not an unread ignore stack.
    write(tmp_path / ".gitignore", "*.log\n")
    write(repo / ".gitignore", "*.tmp\n")
    write(repo / "app.js")
    write(repo / "debug.log")
    write(repo / "scratch.tmp")

    assert names(find_files(repo), repo) == {"app.js", "debug.log"}


def test_no_ignore_dot_keeps_files_only_dot_ignore_excluded(repo):
    write(repo / ".ignore", "*.log\n")
    write(repo / ".gitignore", "*.tmp\n")
    write(repo / "app.js")
    write(repo / "debug.log")
    write(repo / "scratch.tmp")

    assert names(find_files(repo, respect_dot_ignore=False), repo) == {
        "app.js",
        "debug.log",
    }


def test_no_ignore_also_disables_dot_ignore(repo):
    write(repo / ".ignore", "*.log\n")
    write(repo / "app.js")
    write(repo / "debug.log")

    assert names(find_files(repo), repo) == {"app.js"}
    assert names(find_files(repo, respect_gitignore=False), repo) == {
        "app.js",
        "debug.log",
    }


def test_gitignore_outside_a_repository_has_no_effect(tmp_path):
    # `.ignore` is ripgrep's own and needs no repository, so `scratch.tmp` going
    # missing is what proves ignore files are being read here at all.
    plain = tmp_path / "plain"
    write(plain / ".gitignore", "*.log\n")
    write(plain / ".ignore", "*.tmp\n")
    write(plain / "app.js")
    write(plain / "debug.log")
    write(plain / "scratch.tmp")

    assert names(find_files(plain), plain) == {"app.js", "debug.log"}


def test_dot_ignore_outside_a_repository_still_applies(tmp_path):
    plain = tmp_path / "plain"
    write(plain / ".ignore", "*.log\n")
    write(plain / "app.js")
    write(plain / "debug.log")

    assert names(find_files(plain), plain) == {"app.js"}


def test_core_excludes_file_does_not_apply_outside_a_repository(
    repo, tmp_path, isolated_git_env
):
    """One excludes file, two walks: it bites inside a repository and not outside one.

    Walking only the plain directory cannot tell "outside a repository" apart from
    "the excludes file was never read", so the repository half runs in the same test.
    """
    write(isolated_git_env / ".config" / "git" / "ignore", "*.swp\n")
    plain = tmp_path / "plain"
    write(plain / "notes.txt")
    write(plain / "notes.swp")
    write(repo / "notes.txt")
    write(repo / "notes.swp")

    assert names(find_files(plain), plain) == {"notes.txt", "notes.swp"}
    assert names(find_files(repo), repo) == {"notes.txt"}


def test_a_nested_repository_contributes_its_own_info_exclude(repo):
    nested = repo / "vendor" / "lib"
    nested.mkdir(parents=True)
    run_git(nested, "init", "-q")
    write(nested / ".git" / "info" / "exclude", "target/\n")
    write(nested / "src.rs")
    write(nested / "target" / "build.rs")

    assert names(find_files(repo), repo) == {"vendor/lib/src.rs"}


def test_symlinks_are_not_followed_or_listed(repo):
    write(repo / "real.txt")
    write(repo / "tree" / "inner.txt")
    (repo / "link.txt").symlink_to(repo / "real.txt")
    (repo / "link_dir").symlink_to(repo / "tree")

    assert names(find_files(repo), repo) == {"real.txt", "tree/inner.txt"}
    assert names(find_files(repo, recursive=False), repo) == {"real.txt"}


def test_hidden_files_and_directories_are_skipped_by_default(repo):
    write(repo / "visible.txt")
    write(repo / ".env")
    write(repo / ".github" / "workflows" / "ci.yml")

    assert names(find_files(repo), repo) == {"visible.txt"}


def test_hidden_brings_back_every_dotted_path_including_gitignore(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "visible.txt")
    write(repo / ".env")
    write(repo / ".github" / "workflows" / "ci.yml")
    write(repo / "debug.log")

    found = names(find_files(repo, include_hidden=True), repo)
    assert {name for name in found if not name.startswith(".git/")} == {
        ".gitignore",
        ".env",
        ".github/workflows/ci.yml",
        "visible.txt",
    }


def test_the_git_directory_is_skipped_by_default_for_being_hidden(repo):
    """`.git` goes by the dot-prefix rule, so `--hidden` brings it back like `.other`.

    Asserting only the default listing cannot see the *for being hidden*: a walk that
    skipped `.git` by name would pass it just as well. `.other` is the control on one
    side and `include_hidden=True` is the control on the other.
    """
    write(repo / "visible.txt")
    write(repo / ".other" / "x.txt")

    assert names(find_files(repo), repo) == {"visible.txt"}

    found = names(find_files(repo, include_hidden=True), repo)
    assert {".git/HEAD", ".other/x.txt"} <= found


def test_hidden_walks_the_git_directory_like_any_other_dotted_path(repo):
    write(repo / "visible.txt")

    found = names(find_files(repo, include_hidden=True), repo)
    assert {".git/HEAD", ".git/config"} <= found


def test_an_explicitly_named_hidden_directory_is_still_walked(repo):
    write(repo / ".github" / "workflows" / "ci.yml")

    found = find_files(repo / ".github")
    assert names(found, repo) == {".github/workflows/ci.yml"}


def test_an_outer_repositorys_gitignore_stops_at_a_nested_checkout(repo):
    """Git resolves a path against the repository it is in, and ripgrep agrees.

    #91 is this case: a checkout parked inside another one was having the outer
    repository's rules applied to it.
    """
    write(repo / ".gitignore", "*.log\nsecret.txt\n")
    write(repo / "top.log")
    nested = repo / "loadout"
    nested.mkdir()
    run_git(nested, "init", "-q")
    for name in ("inner.log", "secret.txt", "fine.txt"):
        write(nested / name)

    assert names(find_files(repo), repo) == {
        "loadout/fine.txt",
        "loadout/inner.log",
        "loadout/secret.txt",
    }


def test_an_outer_info_exclude_stops_at_a_nested_checkout(repo):
    write(repo / ".git" / "info" / "exclude", "excluded.txt\n")
    write(repo / "excluded.txt")
    write(repo / "kept.txt")
    nested = repo / "vendor"
    nested.mkdir()
    run_git(nested, "init", "-q")
    write(nested / "excluded.txt")

    assert names(find_files(repo), repo) == {"kept.txt", "vendor/excluded.txt"}


def test_the_global_excludes_file_still_applies_inside_a_nested_checkout(
    repo, isolated_git_env
):
    """Dropping the outer repository's rules must not drop git's global ones too."""
    write(isolated_git_env / ".config" / "git" / "ignore", "*.swp\n")
    write(repo / "notes.swp")
    nested = repo / "vendor"
    nested.mkdir()
    run_git(nested, "init", "-q")
    write(nested / "inner.swp")
    write(nested / "inner.txt")

    assert names(find_files(repo), repo) == {"vendor/inner.txt"}


@pytest.mark.parametrize("dot_file", [".ignore", ".rgignore"])
def test_dot_ignore_files_cross_a_nested_checkout_boundary(repo, dot_file):
    """.ignore and .rgignore are ripgrep's, not git's, so no repository bounds them."""
    write(repo / dot_file, "*.skip\n")
    write(repo / "a.skip")
    nested = repo / "inner"
    nested.mkdir()
    run_git(nested, "init", "-q")
    write(nested / "b.skip")
    write(nested / "b.txt")

    assert names(find_files(repo), repo) == {"inner/b.txt"}


def test_a_dot_ignore_reaches_into_a_nested_checkout_where_a_gitignore_stops(tmp_path):
    """The asymmetry the README's `.ignore` recipe rests on, asserted as one claim.

    Each half is pinned a few tests above, but neither says they disagree, which is
    the whole reason for adding a second file rather than renaming the first: the
    same pattern in a `.gitignore` and in an `.ignore` gives different file sets.
    """
    found = {}
    for ignore_file in (".gitignore", ".ignore"):
        root = tmp_path / ignore_file.lstrip(".")
        root.mkdir()
        run_git(root, "init", "-q")
        write(root / ignore_file, "*.skip\n")
        write(root / "outer.skip")
        write(root / "outer.txt")
        nested = root / "inner"
        nested.mkdir()
        run_git(nested, "init", "-q")
        write(nested / "inner.skip")
        write(nested / "inner.txt")
        found[ignore_file] = names(find_files(root), root)

    assert found[".gitignore"] != found[".ignore"]
    assert found[".gitignore"] - found[".ignore"] == {"inner/inner.skip"}
    assert found[".gitignore"] == {"outer.txt", "inner/inner.skip", "inner/inner.txt"}
    assert found[".ignore"] == {"outer.txt", "inner/inner.txt"}


def test_info_exclude_is_found_through_a_linked_worktrees_gitdir_file(repo, tmp_path):
    """A worktree's `.git` is a file, and its info/exclude is the main one's."""
    write(repo / "seed.txt")
    run_git(repo, "add", "-A")
    run_git(repo, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "seed")
    write(repo / ".git" / "info" / "exclude", "excluded.txt\n")
    worktree = tmp_path / "linked"
    run_git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    write(worktree / "excluded.txt")
    write(worktree / "kept.txt")

    assert names(find_files(worktree), worktree) == {"kept.txt", "seed.txt"}


# Every file git points toko at is read the same way: undecodable bytes are replaced
# and the walk goes on without whatever rules that file named. ripgrep reports each of
# these on stderr and still lists the tree; decoding any of them strictly instead ends
# the run, so the caller loses every count over one byte in one metadata file.
def test_a_non_utf8_gitignore_line_does_not_kill_the_walk(repo):
    (repo / ".gitignore").write_bytes(b"ignored.txt\nbad\xff\xfename.txt\n")
    write(repo / "ignored.txt")
    write(repo / "kept.txt")

    assert names(find_files(repo), repo) == {"kept.txt"}


def test_a_non_utf8_gitdir_pointer_does_not_kill_the_walk(tmp_path):
    """And the rules that gitdir named are dropped, not resolved to it anyway.

    The pointer names a real gitdir holding a real rule and then one undecodable byte,
    so `excluded.txt` coming back is the whole claim: replacing the byte leaves a path
    that cannot exist, where discarding it would land on the excludes file after all.
    """
    real = tmp_path / "realrepo" / ".git"
    write(real / "info" / "exclude", "excluded.txt\n")
    worktree = tmp_path / "linked"
    worktree.mkdir()
    (worktree / ".git").write_bytes(b"gitdir: " + bytes(real) + b"\xff\n")
    write(worktree / "excluded.txt")
    write(worktree / "kept.txt")

    assert names(find_files(worktree), worktree) == {"excluded.txt", "kept.txt"}


def test_a_non_utf8_commondir_does_not_kill_the_walk(repo, tmp_path):
    """And the common directory it named is dropped along with its exclude file.

    Same fixture as the linked-worktree test above, where `excluded.txt` is dropped
    through a working `commondir`; here the pointer gains one undecodable byte and
    `excluded.txt` coming back is what tells "dropped" apart from "resolved anyway".
    """
    write(repo / "seed.txt")
    run_git(repo, "add", "-A")
    run_git(repo, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "seed")
    write(repo / ".git" / "info" / "exclude", "excluded.txt\n")
    worktree = tmp_path / "linked"
    run_git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    commondir = repo / ".git" / "worktrees" / "linked" / "commondir"
    commondir.write_bytes(bytes(repo / ".git") + b"\xff\n")
    write(worktree / "excluded.txt")
    write(worktree / "kept.txt")

    assert names(find_files(worktree), worktree) == {
        "excluded.txt",
        "kept.txt",
        "seed.txt",
    }


def test_fifos_and_sockets_are_not_files_to_count(repo):
    """`rg --files` lists none of these, and read_text() on a FIFO never returns."""
    write(repo / "real.txt")
    os.mkfifo(repo / "pipe")
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(repo / "sock"))
        assert names(find_files(repo), repo) == {"real.txt"}
        assert names(find_files(repo, recursive=False), repo) == {"real.txt"}


@pytest.mark.skipif(
    not Path("/dev/zero").exists(), reason="no character device to point at"
)
def test_a_symlinked_device_node_is_not_counted_even_when_following(repo):
    """Reading /dev/zero returns bytes until the machine runs out of them.

    `goodlink.txt` has to come back, because a device node is equally absent from a
    walk that never follows a symlink at all -- which is not what this rules out.
    """
    write(repo / "real.txt")
    (repo / "goodlink.txt").symlink_to(repo / "real.txt")
    (repo / "zerolink").symlink_to("/dev/zero")

    assert names(find_files(repo, follow_symlinks=True), repo) == {
        "real.txt",
        "goodlink.txt",
    }


def test_a_fixture_repo_is_never_the_ambient_one(repo):
    """Fixture git commands must not reach the repository the suite is running in.

    GIT_DIR and friends override directory-based discovery, and git exports GIT_DIR
    when it runs a hook -- so with lefthook's pre-commit hook running this suite, a
    leaked GIT_DIR sends every fixture `git commit` into the developer's own checkout
    and rewrites its HEAD and index. The isolation fixture clears them; this pins that.
    """
    for redirect in GIT_LOCATION_VARS:
        assert redirect not in os.environ, f"{redirect} would redirect fixture git"

    resolved = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert Path(resolved).parent == repo.resolve()


def test_a_fixture_git_command_cannot_omit_its_target_directory():
    """One call site that forgot `cwd=` wrote into the real repository's config.

    Scrubbing the environment does not catch that: a git command with no target at
    all runs wherever the test process happens to be, which under pytest is the
    checkout. So the target is positional and required, and omitting it fails.
    """
    target = next(iter(inspect.signature(run_git).parameters.values()))

    assert target.default is inspect.Parameter.empty
    assert target.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    with pytest.raises(TypeError, match="target"):
        inspect.signature(run_git).bind()


def test_a_leaked_git_dir_does_not_redirect_a_fixture_git(repo, tmp_path, monkeypatch):
    """The helper scrubs for itself, rather than trusting an autouse fixture elsewhere."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    run_git(decoy, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))

    assert (
        Path(run_git(repo, "rev-parse", "--absolute-git-dir").strip())
        == (repo / ".git").resolve()
    )


def test_ambient_config_variables_do_not_reach_a_fixture_git(repo, monkeypatch):
    """GIT_CONFIG_COUNT bypasses GIT_CONFIG_NOSYSTEM and GIT_CONFIG_GLOBAL alike.

    This environment exports it, so a fixture git otherwise inherits whatever
    `url.*.insteadOf` rewrites the ambient configuration carries.
    """
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.https://ambient.example/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")

    listed = run_git(repo, "config", "--list")

    assert "ambient.example" not in listed


def test_every_config_variable_git_reads_from_the_environment_is_scrubbed():
    numbered = {f"GIT_CONFIG_KEY_{n}" for n in range(3)} | {
        f"GIT_CONFIG_VALUE_{n}" for n in range(3)
    }
    ambient = {
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_PARAMETERS": "'x.y=z'",
        **dict.fromkeys(numbered, "v"),
    }

    assert not set(fixture_git_env({**ambient, "PATH": "/usr/bin"})) & set(ambient)
