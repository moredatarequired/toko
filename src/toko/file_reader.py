"""File and directory reading utilities."""

import os
from pathlib import Path

import httpx
import pathspec


def read_gitignore(directory: Path) -> pathspec.PathSpec | None:
    """Read .gitignore file and return a PathSpec.

    Args:
        directory: Directory to look for .gitignore

    Returns:
        PathSpec if .gitignore exists, None otherwise
    """
    gitignore_path = directory / ".gitignore"
    if not gitignore_path.exists():
        return None

    with gitignore_path.open() as f:
        patterns = f.read().splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _build_spec(patterns: list[str] | None) -> pathspec.PathSpec | None:
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _path_relative_to(root: Path, base: Path) -> str:
    return str(root.relative_to(base))


def _should_skip(
    file_path: Path,
    *,
    base_dir: Path,
    gitignore_spec: pathspec.PathSpec | None,
    exclude_spec: pathspec.PathSpec | None,
) -> bool:
    relative = _path_relative_to(file_path, base_dir)
    return bool(
        (gitignore_spec and gitignore_spec.match_file(relative))
        or (exclude_spec and exclude_spec.match_file(relative))
    )


def _prune_directories(
    dirs: list[str],
    *,
    current_root: Path,
    base_dir: Path,
    gitignore_spec: pathspec.PathSpec | None,
) -> None:
    if not gitignore_spec:
        return

    keep: list[str] = []
    for directory in dirs:
        candidate = (current_root / directory).relative_to(base_dir)
        if not gitignore_spec.match_file(f"{candidate}/"):
            keep.append(directory)
    dirs[:] = keep


def _iter_recursive_files(
    base_dir: Path,
    *,
    gitignore_spec: pathspec.PathSpec | None,
    exclude_spec: pathspec.PathSpec | None,
) -> list[Path]:
    discovered: list[Path] = []
    for root, dirs, filenames in os.walk(base_dir):
        root_path = Path(root)
        _prune_directories(
            dirs,
            current_root=root_path,
            base_dir=base_dir,
            gitignore_spec=gitignore_spec,
        )

        for filename in filenames:
            if filename == ".gitignore":
                continue
            candidate = root_path / filename
            if _should_skip(
                candidate,
                base_dir=base_dir,
                gitignore_spec=gitignore_spec,
                exclude_spec=exclude_spec,
            ):
                continue
            discovered.append(candidate)
    return discovered


def _iter_shallow_files(
    base_dir: Path,
    *,
    gitignore_spec: pathspec.PathSpec | None,
    exclude_spec: pathspec.PathSpec | None,
) -> list[Path]:
    discovered: list[Path] = []
    for item in base_dir.iterdir():
        if not item.is_file():
            continue
        if item.name == ".gitignore":
            continue
        if _should_skip(
            item,
            base_dir=base_dir,
            gitignore_spec=gitignore_spec,
            exclude_spec=exclude_spec,
        ):
            continue
        discovered.append(item)
    return discovered


def find_files(
    path: Path,
    *,
    recursive: bool = True,
    respect_gitignore: bool = True,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Find files under a path, optionally recursing and applying skip rules."""
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Single file
    if path.is_file():
        return [path]

    # Directory
    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")

    # Build exclusion spec
    exclude_spec = _build_spec(exclude_patterns)

    # Build gitignore spec if needed
    gitignore_spec = read_gitignore(path) if respect_gitignore else None

    files = (
        _iter_recursive_files(
            path, gitignore_spec=gitignore_spec, exclude_spec=exclude_spec
        )
        if recursive
        else _iter_shallow_files(
            path, gitignore_spec=gitignore_spec, exclude_spec=exclude_spec
        )
    )

    return sorted(files)


def read_file(path: Path) -> str:
    """Read file contents as text.

    Args:
        path: Path to file

    Returns:
        File contents as string

    Raises:
        UnicodeDecodeError: If file is not valid UTF-8
    """
    return path.read_text()


def fetch_url(url: str, *, timeout: float = 30.0) -> str:
    """Fetch content from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Content as string

    Raises:
        httpx.HTTPError: If request fails
        UnicodeDecodeError: If response is not valid UTF-8
    """
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text
