"""File and directory reading utilities."""

import os
from pathlib import Path

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


def find_files(
    path: Path,
    *,
    recursive: bool = True,
    respect_gitignore: bool = True,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Find all files in a path, respecting .gitignore and exclude patterns.

    Args:
        path: Path to file or directory
        recursive: If True, recurse into subdirectories
        respect_gitignore: If True, respect .gitignore files
        exclude_patterns: List of glob patterns to exclude

    Returns:
        List of file paths

    Raises:
        FileNotFoundError: If path doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Single file
    if path.is_file():
        return [path]

    # Directory
    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")

    # Build exclusion spec
    exclude_spec = None
    if exclude_patterns:
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)

    # Build gitignore spec if needed
    gitignore_spec = None
    if respect_gitignore:
        gitignore_spec = read_gitignore(path)

    # Find all files
    files = []
    if recursive:
        # Recursive walk
        for root, dirs, filenames in os.walk(path):
            root_path = Path(root)

            # Check gitignore for directories (modify dirs in-place to skip)
            if gitignore_spec:
                # Get relative paths for matching
                rel_dirs = [str((root_path / d).relative_to(path)) + "/" for d in dirs]
                matched_dirs = gitignore_spec.match_files(rel_dirs)
                # Remove matched directories from dirs to skip them
                dirs[:] = [
                    d
                    for d, rel in zip(dirs, rel_dirs, strict=False)
                    if rel not in matched_dirs
                ]

            # Add files from this directory
            for filename in filenames:
                # Skip .gitignore itself
                if filename == ".gitignore":
                    continue

                file_path = root_path / filename
                rel_path = str(file_path.relative_to(path))

                # Check gitignore
                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    continue

                # Check exclude patterns
                if exclude_spec and exclude_spec.match_file(rel_path):
                    continue

                files.append(file_path)
    else:
        # Non-recursive - just list files in directory
        for item in path.iterdir():
            if not item.is_file():
                continue

            # Skip .gitignore itself
            if item.name == ".gitignore":
                continue

            rel_path = str(item.relative_to(path))

            # Check gitignore
            if gitignore_spec and gitignore_spec.match_file(rel_path):
                continue

            # Check exclude patterns
            if exclude_spec and exclude_spec.match_file(rel_path):
                continue

            files.append(item)

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
