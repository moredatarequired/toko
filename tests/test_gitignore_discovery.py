"""Ignore-file discovery: upward search, per-directory stack, git's exclude files."""

import os
import subprocess
from pathlib import Path

import pytest

from toko import file_reader
from toko.file_reader import find_files


@pytest.fixture(autouse=True)
def _isolated_git_env(monkeypatch, tmp_path):
    """Keep every test off the developer's real git config and home directory."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)  # noqa: S603, S607


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "fx"
    root.mkdir()
    git("init", "-q", cwd=root)
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
    git("config", "--local", "core.excludesFile", str(excludes), cwd=repo)
    write(repo / "notes.txt")
    write(repo / "notes.swp")

    assert names(find_files(repo), repo) == {"notes.txt"}


def test_xdg_git_ignore_is_the_default_excludes_file(repo, tmp_path):
    write(tmp_path / "home" / ".config" / "git" / "ignore", "*.swp\n")
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
    git("init", "-q", cwd=nested)
    write(nested / ".gitignore", "target/\n")
    write(nested / "src.rs")
    write(nested / "target" / "build.rs")

    assert names(find_files(repo), repo) == {"vendor/lib/src.rs"}


def test_ignored_directories_are_pruned_rather_than_walked(repo, monkeypatch):
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "loadout" / "app.js")
    for i in range(5):
        write(repo / "loadout" / "node_modules" / f"pkg{i}" / "index.js")

    visited: list[str] = []
    real_walk = os.walk

    def recording_walk(top, *args, **kwargs):
        for root, dirs, filenames in real_walk(top, *args, **kwargs):
            visited.append(root)
            yield root, dirs, filenames

    monkeypatch.setattr(file_reader.os, "walk", recording_walk)
    found = find_files(repo)

    assert names(found, repo) == {"loadout/app.js"}
    assert not [root for root in visited if "node_modules" in root]


def test_no_ignore_keeps_every_ignored_file(repo):
    write(repo / ".gitignore", "*.log\n")
    write(repo / "loadout" / ".gitignore", "node_modules/\n")
    write(repo / "loadout" / "node_modules" / "index.js")
    write(repo / "top.log")
    write(repo / "README.md")

    found = find_files(repo, respect_gitignore=False)
    assert names(found, repo) == {
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
