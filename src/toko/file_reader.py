"""File and directory reading utilities."""

import os
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import httpx
import pathspec

# Reported rather than raised: one broken link, or one directory that cannot be opened,
# should not end a walk that still has the rest of the tree to cover.
ErrorReporter = Callable[[str], None]


class _IgnoreLayer(NamedTuple):
    """One ignore file's patterns, anchored at the directory they are relative to."""

    anchor: str  # absolute path of that directory, with a trailing separator
    spec: pathspec.PathSpec

    def check(self, absolute_path: str, *, is_dir: bool) -> bool | None:
        relative = absolute_path.removeprefix(self.anchor)
        return self.spec.check_file(f"{relative}/" if is_dir else relative).include


class _IgnoreRules(NamedTuple):
    """The ignore files reaching one directory, kept apart by the kind they came from.

    ripgrep ranks them in two dimensions at once: `.ignore` and `.rgignore` outrank
    `.gitignore` and `.git/info/exclude` however deep either one sits, and only within a
    single kind does the deeper file win. One list ordered by depth can express the
    second rule but not the first, so a root `.rgignore` would lose to a nested
    `.gitignore` that ripgrep has it beating.
    """

    git: tuple[_IgnoreLayer, ...] = ()
    dot: tuple[_IgnoreLayer, ...] = ()

    def matches(self, absolute_path: str, *, is_dir: bool = False) -> bool:
        for group in (self.dot, self.git):
            for layer in reversed(group):
                matched = layer.check(absolute_path, is_dir=is_dir)
                if matched is not None:
                    return matched
        return False


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


def _resolve_against(base: Path, target: str) -> Path:
    path = Path(target)
    return path if path.is_absolute() else (base / path).resolve()


def _git_dir(directory: Path) -> Path | None:
    """Locate this directory's git metadata, returning None if it holds none.

    A linked worktree and a submodule have a `.git` *file* naming the real gitdir
    instead of a directory, and a worktree's `info/exclude` is shared: it sits in the
    common directory that gitdir's `commondir` points at, not in the per-worktree one.
    toko is itself developed in worktrees, so treating the file as an absent `.git`
    silently drops every rule in the checkout's exclude file.
    """
    marker = directory / ".git"
    try:
        marked = marker.stat()
    except OSError:
        return None
    if stat.S_ISDIR(marked.st_mode):
        return marker
    if not stat.S_ISREG(marked.st_mode):
        return None
    try:
        pointer = marker.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not pointer.startswith("gitdir:"):
        return None
    gitdir = _resolve_against(directory, pointer.removeprefix("gitdir:").strip())
    try:
        shared = (gitdir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return gitdir
    return _resolve_against(gitdir, shared) if shared else gitdir


def _find_repo_root(directory: Path) -> Path | None:
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _is_within(directory: Path, root: Path | None) -> bool:
    return root is not None and (directory == root or root in directory.parents)


# Weakest first: .git/info/exclude loses to the .gitignore beside it, and both lose to
# .ignore/.rgignore, which _IgnoreRules ranks above them whatever depth each sits at.
DOT_IGNORE_FILES = (".ignore", ".rgignore")


def _git_layers(directory: Path, git_dir: Path | None) -> list[_IgnoreLayer]:
    sources = [directory / ".gitignore"]
    if git_dir is not None:
        sources.insert(0, git_dir / "info" / "exclude")
    layers = (_ignore_layer(directory, source) for source in sources)
    return [layer for layer in layers if layer is not None]


def _dot_layers(directory: Path) -> list[_IgnoreLayer]:
    layers = (_ignore_layer(directory, directory / name) for name in DOT_IGNORE_FILES)
    return [layer for layer in layers if layer is not None]


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


def _global_excludes_spec(start: Path) -> pathspec.PathSpec | None:
    """core.excludesFile, read once and re-anchored at each repository it applies to."""
    configured = _git_config_value("core.excludesFile", cwd=start)
    if configured:
        return _read_spec(Path(configured).expanduser())
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return _read_spec(config_home / "git" / "ignore")


def _global_layers(
    repo_root: Path, spec: pathspec.PathSpec | None
) -> tuple[_IgnoreLayer, ...]:
    return () if spec is None else (_IgnoreLayer(_dir_prefix(repo_root), spec),)


class _WalkOptions(NamedTuple):
    base_prefix: str
    recursive: bool
    respect_gitignore: bool
    respect_dot_ignore: bool
    include_hidden: bool
    follow_symlinks: bool
    exclude_spec: pathspec.PathSpec | None
    global_spec: pathspec.PathSpec | None
    report: ErrorReporter


class _Frame(NamedTuple):
    """One directory left to walk, and everything its entries are judged against."""

    path: Path  # spelled the way the caller named the base, so results keep that shape
    prefix: str  # the same directory absolute, with a trailing separator
    rules: _IgnoreRules
    repo_root: Path | None
    # (device, inode) of every directory on the way here, kept only when following
    # symlinks: a link back to one of them is a cycle, and descending would not stop.
    ancestors: tuple[tuple[tuple[int, int], Path], ...] = ()


def _should_skip(absolute_path: str, frame: _Frame, options: _WalkOptions) -> bool:
    if frame.rules.matches(absolute_path):
        return True
    relative = absolute_path.removeprefix(options.base_prefix)
    return bool(options.exclude_spec and options.exclude_spec.match_file(relative))


def _describe(path: str, error: OSError) -> str:
    return f"{path}: {error.strerror or error}"


def _initial_frame(base_dir: Path, base_abs: Path, options: _WalkOptions) -> _Frame:
    """Build the base frame: every ignore file above the base that still reaches it."""
    repo_root = _find_repo_root(base_abs) if options.respect_gitignore else None
    git: list[_IgnoreLayer] = []
    dot: list[_IgnoreLayer] = []
    if options.respect_gitignore:
        if repo_root is not None:
            git.extend(_global_layers(repo_root, options.global_spec))
        for directory in (*reversed(base_abs.parents), base_abs):
            if _is_within(directory, repo_root):
                git.extend(_git_layers(directory, _git_dir(directory)))
            # .ignore and .rgignore are ripgrep's own, not git's, so a repository
            # boundary does not stop them the way it stops the git files.
            if options.respect_dot_ignore:
                dot.extend(_dot_layers(directory))
    ancestors: tuple[tuple[tuple[int, int], Path], ...] = ()
    if options.follow_symlinks:
        try:
            info = base_abs.stat()
        except OSError:
            info = None
        if info is not None:
            ancestors = (((info.st_dev, info.st_ino), base_dir),)
    return _Frame(
        base_dir,
        options.base_prefix,
        _IgnoreRules(tuple(git), tuple(dot)),
        repo_root,
        ancestors,
    )


def _descend(parent: _Frame, name: str, options: _WalkOptions) -> _Frame:
    """Build a subdirectory's frame, stacking its own ignore files onto its parent's."""
    prefix = parent.prefix + name + os.sep
    directory = Path(prefix.rstrip(os.sep))
    repo_root = parent.repo_root
    git = parent.rules.git
    dot = parent.rules.dot
    if options.respect_gitignore:
        git_dir = _git_dir(directory)
        if git_dir is not None:
            # A checkout nested inside another one answers to its own git ignore files
            # and no further: git resolves a path against the repository it belongs to,
            # and ripgrep follows it, so the outer rules are dropped rather than kept.
            repo_root = directory
            git = _global_layers(directory, options.global_spec)
        if repo_root is not None:
            git = (*git, *_git_layers(directory, git_dir))
        if options.respect_dot_ignore:
            dot = (*dot, *_dot_layers(directory))
    return _Frame(parent.path / name, prefix, _IgnoreRules(git, dot), repo_root)


class _Entry(NamedTuple):
    is_dir: bool
    is_file: bool
    ident: tuple[int, int] | None  # (device, inode), resolved only when following


def _inspect(entry: os.DirEntry[str], options: _WalkOptions) -> _Entry | None:
    """Classify one directory entry, returning None if it is not one to walk into."""
    if entry.is_symlink():
        if not options.follow_symlinks:
            return None
        # stat() resolves the link, and says so when it dangles or spirals; lstat
        # would only ever report the link itself, which always exists.
        info = entry.stat()
        is_dir = stat.S_ISDIR(info.st_mode)
        return _Entry(is_dir, stat.S_ISREG(info.st_mode), _identity(info, is_dir))
    is_dir = entry.is_dir(follow_symlinks=False)
    ident = None
    if is_dir and options.follow_symlinks:
        ident = _identity(entry.stat(follow_symlinks=False), is_dir=True)
    return _Entry(is_dir, entry.is_file(follow_symlinks=False), ident)


def _identity(info: os.stat_result, is_dir: bool) -> tuple[int, int] | None:
    return (info.st_dev, info.st_ino) if is_dir else None


def _looped_ancestor(frame: _Frame, ident: tuple[int, int] | None) -> Path | None:
    if ident is None:
        return None
    return next((path for seen, path in frame.ancestors if seen == ident), None)


def _walk(base_dir: Path, options: _WalkOptions) -> list[Path]:
    discovered: list[Path] = []
    stack = [_initial_frame(base_dir, base_dir.absolute(), options)]
    while stack:
        frame = stack.pop()
        try:
            entries = list(os.scandir(frame.path))
        except OSError as error:
            options.report(_describe(str(frame.path), error))
            continue
        for entry in entries:
            if not options.include_hidden and entry.name.startswith("."):
                continue
            try:
                found = _inspect(entry, options)
            except OSError as error:
                options.report(_describe(entry.path, error))
                continue
            if found is None:
                continue
            absolute = frame.prefix + entry.name
            if found.is_dir:
                if not options.recursive or frame.rules.matches(absolute, is_dir=True):
                    continue
                ancestor = _looped_ancestor(frame, found.ident)
                if ancestor is not None:
                    options.report(
                        f"File system loop found: {entry.path} points to an "
                        f"ancestor {ancestor}"
                    )
                    continue
                child = _descend(frame, entry.name, options)
                if found.ident is not None:
                    child = child._replace(
                        ancestors=(*frame.ancestors, (found.ident, child.path))
                    )
                stack.append(child)
            elif found.is_file and not _should_skip(absolute, frame, options):
                discovered.append(frame.path / entry.name)
            # Anything else -- a FIFO, a socket, a device node -- is not a file to
            # count, and `rg --files` does not list one either. Reading a FIFO would
            # block the whole run until something opened the far end.
    return discovered


def find_files(
    path: Path,
    *,
    recursive: bool = True,
    respect_gitignore: bool = True,
    respect_dot_ignore: bool = True,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
    exclude_patterns: list[str] | None = None,
    on_error: ErrorReporter | None = None,
) -> list[Path]:
    """Find files under a path, optionally recursing and applying skip rules.

    `on_error` receives the links and directories the walk could not resolve. They are
    reported rather than raised so that one of them cannot cost the caller the counts
    for every other file in the tree.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # A path named on the command line is followed whether or not follow_symlinks is
    # set, matching ripgrep: the flag governs links met while walking, not arguments.
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")

    options = _WalkOptions(
        base_prefix=_dir_prefix(path.absolute()),
        recursive=recursive,
        respect_gitignore=respect_gitignore,
        respect_dot_ignore=respect_gitignore and respect_dot_ignore,
        include_hidden=include_hidden,
        follow_symlinks=follow_symlinks,
        exclude_spec=_build_spec(exclude_patterns),
        global_spec=_global_excludes_spec(path) if respect_gitignore else None,
        report=on_error if on_error is not None else lambda _message: None,
    )
    return sorted(_walk(path, options))


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
