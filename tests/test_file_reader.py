"""Tests for file_reader module."""

import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from toko.file_reader import fetch_url, find_files, read_file, read_gitignore


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


@pytest.mark.parametrize("respect_gitignore", [True, False])
def test_find_files_skips_dot_git(respect_gitignore):
    """.git is the repo boundary, so no .gitignore setting can bring it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        (tmpdir_path / "file.txt").write_text("test")
        git_dir = tmpdir_path / ".git" / "objects"
        git_dir.mkdir(parents=True)
        (tmpdir_path / ".git" / "config").write_text("[core]")
        (git_dir / "abc").write_text("object")

        files = find_files(tmpdir_path, respect_gitignore=respect_gitignore)
        assert [f.name for f in files] == ["file.txt"]


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
