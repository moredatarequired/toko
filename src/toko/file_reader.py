"""File and directory reading utilities."""

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

import httpx
import pathspec


class _IgnoreLayer(NamedTuple):
    """One ignore file's patterns, anchored at the directory they are relative to."""

    anchor: str  # absolute path of that directory, with a trailing separator
    spec: pathspec.PathSpec

    def check(self, absolute_path: str, *, is_dir: bool) -> bool | None:
        relative = absolute_path.removeprefix(self.anchor)
        return self.spec.check_file(f"{relative}/" if is_dir else relative).include


def _dir_prefix(directory: Path) -> str:
    return str(directory).rstrip(os.sep) + os.sep


def _build_spec(patterns: list[str] | None) -> pathspec.PathSpec | None:
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _read_spec(path: Path) -> pathspec.PathSpec | None:
    try:
        patterns = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    return _build_spec(patterns)


def read_gitignore(directory: Path) -> pathspec.PathSpec | None:
    """Read one directory's own .gitignore, without looking above or below it."""
    return _read_spec(directory / ".gitignore")


def _ignore_layer(anchor: Path, ignore_file: Path) -> _IgnoreLayer | None:
    spec = _read_spec(ignore_file)
    if spec is None:
        return None
    return _IgnoreLayer(_dir_prefix(anchor), spec)


def _find_repo_root(directory: Path) -> Path | None:
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_config_value(key: str, *, cwd: Path) -> str | None:
    """Ask git for a config value, treating a missing or broken git as unset."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "config", "--get", key],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _global_excludes_file(repo_root: Path) -> Path:
    configured = _git_config_value("core.excludesFile", cwd=repo_root)
    if configured:
        return Path(configured).expanduser()
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return config_home / "git" / "ignore"


def _inherited_ignore_layers(directory: Path) -> list[_IgnoreLayer]:
    """Ignore rules reaching `directory` from the repository it sits in.

    Ordered weakest first: git resolves a path against the last pattern that
    matches it, so a deeper .gitignore has to come after a shallower one.
    """
    repo_root = _find_repo_root(directory)
    if repo_root is None:
        return []

    between: list[Path] = []
    ancestor = directory
    while ancestor != repo_root and ancestor != ancestor.parent:
        ancestor = ancestor.parent
        between.append(ancestor)

    sources = [
        (repo_root, _global_excludes_file(repo_root)),
        (repo_root, repo_root / ".git" / "info" / "exclude"),
        *((d, d / ".gitignore") for d in reversed(between)),
    ]
    layers = (_ignore_layer(anchor, source) for anchor, source in sources)
    return [layer for layer in layers if layer is not None]


def _is_ignored(
    absolute_path: str, layers: list[_IgnoreLayer], *, is_dir: bool = False
) -> bool:
    for layer in reversed(layers):
        matched = layer.check(absolute_path, is_dir=is_dir)
        if matched is not None:
            return matched
    return False


def _should_skip(
    absolute_path: str,
    *,
    base_prefix: str,
    layers: list[_IgnoreLayer],
    exclude_spec: pathspec.PathSpec | None,
) -> bool:
    if _is_ignored(absolute_path, layers):
        return True
    relative = absolute_path.removeprefix(base_prefix)
    return bool(exclude_spec and exclude_spec.match_file(relative))


def _iter_recursive_files(
    base_dir: Path, *, respect_gitignore: bool, exclude_spec: pathspec.PathSpec | None
) -> list[Path]:
    base_abs = base_dir.absolute()
    base_prefix = _dir_prefix(base_abs)
    inherited = _inherited_ignore_layers(base_abs) if respect_gitignore else []
    layers_by_dir: dict[Path, list[_IgnoreLayer]] = {}

    discovered: list[Path] = []
    for root, dirs, filenames in os.walk(base_dir):
        root_path = Path(root)
        root_abs = root_path.absolute()
        root_prefix = _dir_prefix(root_abs)
        layers = layers_by_dir.get(root_abs.parent, inherited)
        own_layer = (
            _ignore_layer(root_abs, root_path / ".gitignore")
            if respect_gitignore
            else None
        )
        if own_layer:
            layers = [*layers, own_layer]
        layers_by_dir[root_abs] = layers

        # No .gitignore can exclude .git: git treats it as the repo boundary
        # rather than an ignorable path, so prune it however we were configured.
        if ".git" in dirs:
            dirs.remove(".git")
        dirs[:] = [
            name
            for name in dirs
            if not _is_ignored(root_prefix + name, layers, is_dir=True)
        ]

        for filename in filenames:
            if filename == ".gitignore":
                continue
            if _should_skip(
                root_prefix + filename,
                base_prefix=base_prefix,
                layers=layers,
                exclude_spec=exclude_spec,
            ):
                continue
            discovered.append(root_path / filename)
    return discovered


def _iter_shallow_files(
    base_dir: Path, *, respect_gitignore: bool, exclude_spec: pathspec.PathSpec | None
) -> list[Path]:
    base_abs = base_dir.absolute()
    base_prefix = _dir_prefix(base_abs)
    layers: list[_IgnoreLayer] = []
    if respect_gitignore:
        layers = _inherited_ignore_layers(base_abs)
        own_layer = _ignore_layer(base_abs, base_dir / ".gitignore")
        if own_layer:
            layers.append(own_layer)

    discovered: list[Path] = []
    for item in base_dir.iterdir():
        if not item.is_file():
            continue
        if item.name == ".gitignore":
            continue
        if _should_skip(
            base_prefix + item.name,
            base_prefix=base_prefix,
            layers=layers,
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

    exclude_spec = _build_spec(exclude_patterns)

    files = (
        _iter_recursive_files(
            path, respect_gitignore=respect_gitignore, exclude_spec=exclude_spec
        )
        if recursive
        else _iter_shallow_files(
            path, respect_gitignore=respect_gitignore, exclude_spec=exclude_spec
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
