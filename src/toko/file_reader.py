"""File and directory reading utilities."""

import functools
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


# pathspec's own name for the separator that opened a match on something *below* a
# pattern; it reads the same group in GitIgnoreSpec to rank a directory match.
_DESCENDANT_MARK = "ps_d"


def _pathspec_internals_are_live() -> bool:
    pattern = next(
        iter(pathspec.PathSpec.from_lines("gitwildmatch", ["dir/"]).patterns)
    )
    found = pattern.match_file("dir/below.txt")
    if found is None or found.match.groupdict().get(_DESCENDANT_MARK) is None:
        return False
    # The group existing is not the whole requirement: `_matched_self` reads how far
    # it reaches to tell a directory's own match from a match on something inside it,
    # so a release that kept the name and widened the capture would pass a check on
    # the name alone while making every descendant read as the directory itself.
    if (
        _matched_self(pattern, "dir/below.txt") is not None
        or _matched_self(pattern, "dir/") is not True
    ):
        return False
    # `_probe` reads a second internal, and one the group check cannot stand in for:
    # `Pattern.pattern` holds the raw source text, which is the only place the
    # trailing separator marking a rule directory-only survives -- pathspec compiles
    # it away into the regex. A release that stopped carrying the raw text would
    # leave every `dir/` rule probed as a plain name, matching nothing and pruning
    # nothing, with no other symptom.
    return _probe("dir", pattern, is_dir=True) == "dir/"


def _probe(relative: str, pattern: pathspec.Pattern, *, is_dir: bool) -> str:
    """How one entry is spelled for one rule: only a `dir/` rule sees the separator.

    ripgrep judges every entry on its own path and leaves containment to the walk,
    so a directory is `dir` to `dir/**`, which matches only what is inside it, and
    `dir/` to `dir/`, which is the rule that prunes it. Showing the separator to every
    rule instead prunes the directory ripgrep descends into, and with it whatever a
    later negation re-includes underneath -- files dropped from the count in silence,
    since pruning `dir` and excluding its contents otherwise list the same tree.
    """
    source = getattr(pattern, "pattern", None)
    # pathspec compiles its regex from the stripped pattern, so `dir/ ` is the very
    # same rule as `dir/`. Re-deriving directory-only-ness from the raw text has to
    # strip it the same way, or a rule pathspec still reads as directory-only is
    # probed as a plain name, matches nothing, and quietly stops pruning.
    directory_only = isinstance(source, str) and source.rstrip().endswith("/")
    return f"{relative}/" if is_dir and directory_only else relative


def _matched_self(pattern: pathspec.Pattern, probe: str) -> bool | None:
    """One pattern's verdict on the entry `probe` names, never on what lies beneath it.

    Ignore rules govern discovery, not arguments: a path the caller names is searched,
    and the rules still apply to what is found underneath it. ripgrep gets both halves
    from matching each entry against the rule on its own and leaving containment to the
    walk, which prunes a matched directory and so never reaches its children. pathspec
    instead expands `dir/` to match every descendant, standing in for that pruning --
    which is redundant wherever the walk did the pruning itself, and wrong at the one
    directory it never judged, the one the caller named. Reading `dir/x.txt` as a match
    for `dir/` there emptied the walk of a named ignored directory. So the expansion is
    dropped and the walk is left to prune; a rule matching something *within* the tree,
    `*.log` against `dir/y.log`, is the entry's own match and still bites.
    """
    found = pattern.match_file(probe)
    if found is None:
        return None
    if found.match.groupdict().get(_DESCENDANT_MARK) is None:
        return pattern.include
    # A directory is probed with a trailing separator, so a pattern that consumed
    # exactly that one matched the directory itself rather than something inside it.
    if found.match.end(_DESCENDANT_MARK) == len(probe):
        return pattern.include
    return None


# Both internals are private to pathspec and the requirement carries no upper bound,
# so a release that moved either would not raise. Renaming, dropping or widening the
# mark would have `_matched_self` call every descendant match the directory's own, and
# `dir/**` would quietly prune `dir` again; dropping the raw source text would have
# `_probe` spell every directory without its separator, and every `dir/` rule would
# quietly stop pruning. Either one undoes the walk below for an installed user while
# the pinned lockfile keeps CI green, so say so at import instead. The check runs here
# rather than beside the function because it exercises `_matched_self` and `_probe`.
if not _pathspec_internals_are_live():
    raise RuntimeError(
        f"pathspec no longer exposes what toko.file_reader reads of it: a match on a "
        f"descendant through the {_DESCENDANT_MARK!r} capture group, which tells a "
        f"directory's own match from a match on something inside it, and the raw "
        f"pattern text on Pattern.pattern, which is where the trailing separator "
        f"marking a rule directory-only survives."
    )


class _Anchor(NamedTuple):
    """How a walked path is spelled for one ignore file: strip `prefix`, or lead a name.

    For an ignore file in a directory the walk reached, `prefix` is that directory and
    `lead` is empty -- every path the walk produces starts with it, so stripping the
    prefix leaves the path the rules are written against.

    An ignore file *above* the walk root is spelled for differently, and not the way
    git would spell it. ripgrep holds one path for such a file -- the walk root,
    resolved, named relative to the directory holding it -- and puts each entry to
    those rules as that path joined with the entry's own *name*, dropping whatever
    directories lie between the two. Walking `root`, `root/sub/deep.bb` is judged as
    `root/deep.bb`: `/root/*.bb` up there drops it and `/root/sub/deep.bb` up there
    leaves it alone, which is the reverse of git's answer and the reverse of what the
    same two rules do from the walk root's own ignore file. Spelling those layers the
    way git spells them costs parity in both directions at once -- a file ripgrep lists
    dropped from the count, and a file ripgrep ignores counted, read and sent on.

    `lead` carries that path and is set for no other kind of layer, so a non-empty
    `lead` is what marks one as coming from above; `prefix` goes unread there, a name
    having nothing to strip.
    """

    prefix: str  # absolute path of a directory, with a trailing separator
    lead: str = ""  # the walk root resolved, named relative to a directory above it

    def spell(self, absolute_path: str) -> str:
        if self.lead:
            _, _, name = absolute_path.rpartition(os.sep)
            return self.lead + name
        return absolute_path.removeprefix(self.prefix)


class _IgnoreLayer(NamedTuple):
    """One ignore file's patterns, anchored at the directory they are relative to."""

    anchor: _Anchor
    spec: pathspec.PathSpec

    def check(self, absolute_path: str, *, is_dir: bool) -> bool | None:
        relative = self.anchor.spell(absolute_path)
        # Last match wins, the way pathspec and git both rank a file's patterns.
        verdict: bool | None = None
        for pattern in self.spec.patterns:
            if pattern.include is None:
                continue
            matched = _matched_self(pattern, _probe(relative, pattern, is_dir=is_dir))
            if matched is not None:
                verdict = matched
        return verdict


class _IgnoreRules(NamedTuple):
    """The ignore files reaching one directory, kept apart by the kind they came from.

    ripgrep ranks them in two dimensions at once: a stronger kind outranks a weaker one
    however deep either file sits, and only within a single kind does the deeper file
    win. `.rgignore` beats `.ignore`, which beats `.gitignore` and `.git/info/exclude`.
    One list ordered by depth can express the second rule but not the first, so a root
    `.rgignore` would lose to a nested `.ignore` that ripgrep has it beating.
    """

    git: tuple[_IgnoreLayer, ...] = ()
    ignore: tuple[_IgnoreLayer, ...] = ()
    rgignore: tuple[_IgnoreLayer, ...] = ()

    def matches(self, absolute_path: str, *, is_dir: bool = False) -> bool:
        for group in (self.rgignore, self.ignore, self.git):
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
        patterns = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return _build_spec(patterns)


def read_gitignore(directory: Path) -> pathspec.PathSpec | None:
    """Read one directory's own .gitignore, without looking above or below it."""
    return _read_spec(directory / ".gitignore")


def _ignore_layer(anchor: _Anchor, ignore_file: Path) -> _IgnoreLayer | None:
    spec = _read_spec(ignore_file)
    if spec is None:
        return None
    return _IgnoreLayer(anchor, spec)


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
        # Undecodable bytes are replaced rather than round-tripped through os.fsdecode
        # the way _global_config_value does: the path that leaves cannot exist, so the
        # rules it names are dropped, which is what ripgrep does with a gitdir it
        # cannot decode. Decoding strictly instead raised out of a walk ripgrep
        # finishes, costing the caller every count in the tree.
        shared = (
            (gitdir / "commondir").read_text(encoding="utf-8", errors="replace").strip()
        )
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


def _git_layers(
    directory: Path, git_dir: Path | None, anchor: _Anchor
) -> list[_IgnoreLayer]:
    # Weakest first: .git/info/exclude loses to the .gitignore beside it, and both lose
    # to .ignore and .rgignore, which _IgnoreRules ranks above them at any depth.
    sources = [directory / ".gitignore"]
    if git_dir is not None:
        sources.insert(0, git_dir / "info" / "exclude")
    layers = (_ignore_layer(anchor, source) for source in sources)
    return [layer for layer in layers if layer is not None]


def _dot_layer(directory: Path, name: str, anchor: _Anchor) -> tuple[_IgnoreLayer, ...]:
    layer = _ignore_layer(anchor, directory / name)
    return () if layer is None else (layer,)


# Everything git resolves its global scope from: the file GIT_CONFIG_GLOBAL names, the
# homes it falls back to when that is unset, and the GIT_CONFIG_* variables that carry
# config in the environment rather than in a file. The cached lookup is keyed on these
# and on the state of the files themselves, so a caller that repoints any of them, or
# rewrites the file they name, asks git again instead of reading back an answer
# resolved from a configuration that no longer holds.
_GIT_CONFIG_HOME_VARS = ("HOME", "XDG_CONFIG_HOME")


def _git_config_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("GIT_CONFIG") or name in _GIT_CONFIG_HOME_VARS
        )
    )


def _git_config_files() -> tuple[Path, ...]:
    """List git's global scope: GIT_CONFIG_GLOBAL, or the two files in the homes."""
    named = os.environ.get("GIT_CONFIG_GLOBAL")
    if named:
        return (Path(named),)
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return (config_home / "git" / "config", Path.home() / ".gitconfig")


def _git_config_stamp() -> tuple[tuple[str, int, int, int], ...]:
    """Stamp those files by inode, size and modification time, for the cache to key on.

    The inode is in there because `git config --global` rewrites the file through a
    lock file and a rename, which moves it whether or not the new contents are the
    same length or the clock has ticked since the last write.

    An `include.path` inside one of them is not followed, so a run that edits an
    included file and not the file including it still reads back the cached answer.
    """
    try:
        paths = _git_config_files()
    except RuntimeError:  # no home directory to resolve the unset defaults against
        return ()
    stamps = []
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            continue
        stamps.append((str(path), info.st_ino, info.st_size, info.st_mtime_ns))
    return tuple(stamps)


def _global_config_value(key: str) -> str | None:
    return _cached_global_config_value(
        key, _git_config_environment(), _git_config_stamp()
    )


@functools.lru_cache(maxsize=8)
def _cached_global_config_value(
    key: str,
    _environment: tuple[tuple[str, str], ...],
    _stamp: tuple[tuple[str, int, int, int], ...],
) -> str | None:
    """Ask git for a config value from its global scope, treating a broken git as unset.

    Cached because find_files runs this once per path argument, so `toko a b c` was
    paying for three `git config` subprocesses to be told the same thing three times.
    The key carries everything the answer is resolved from -- the environment that
    selects the file, and the file's own size and modification time -- because the
    cache outlives the process's configuration otherwise: a caller that rewrites its
    global config between two walks was answered from the config before it. Bounded
    rather than unbounded, since a key that moves with a file is a key that a
    long-lived caller can go on minting.

    The only value read here is core.excludesFile, which is a path, so git's stdout is
    decoded with os.fsdecode -- exactly how Python decodes every other filesystem path.
    `text=True` would instead use locale.getencoding(): a non-ASCII excludes path under
    a non-UTF-8 locale then raised UnicodeDecodeError straight out of a function whose
    whole contract is to degrade quietly, killing a run over a tree of ASCII files.
    Hardcoding UTF-8 is not enough either -- it decodes to real characters that open()
    cannot re-encode when the filesystem encoding is ASCII. fsdecode round-trips: its
    surrogates encode back to git's original bytes, so the file is still found.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "config", "--global", "--get", key],  # noqa: S607
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return os.fsdecode(result.stdout).strip() or None


def _global_excludes_spec() -> pathspec.PathSpec | None:
    """core.excludesFile, read once and re-anchored at each repository it applies to.

    Only git's global scope is read. ripgrep parses $HOME/.gitconfig and
    $XDG_CONFIG_HOME/git/config itself and never opens a repository's own config, so a
    repo-local core.excludesFile moves what `git status` hides without moving what `rg`
    lists. Reading it here diverged from ripgrep in both directions at once: a local
    file was applied that ripgrep ignores, and a local setting naming a path that does
    not exist suppressed the global file, which for git it does and for ripgrep it
    does not. A configured path that is missing is still not fallen back from, which is
    what both git and ripgrep do.
    """
    configured = _global_config_value("core.excludesFile")
    if configured:
        return _read_spec(Path(configured).expanduser())
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return _read_spec(config_home / "git" / "ignore")


def _global_layers(
    spec: pathspec.PathSpec | None, named: Path
) -> tuple[_IgnoreLayer, ...]:
    """core.excludesFile, anchored where ripgrep anchors it: the working directory.

    This is a deliberate divergence from git, which resolves the file's anchored and
    multi-segment rules against the root of the repository a path belongs to. A rule
    of `/x.txt` names the x.txt in the directory toko was run from, not the one at the
    top of the checkout, and the same excludes file reads to a different answer from a
    subdirectory. See https://github.com/moredatarequired/toko/issues/133.

    ripgrep judges each path the way it spells it -- the walk root as the caller named
    it, with the rest of the tree hung off that -- against a matcher rooted at the
    working directory. A relative name strips against that directory and is matched
    with whatever spelling is left, `..` segments and all: walking `..` is judged as
    `../x.txt`, which an anchored `/x.txt` misses and a `*/x.txt` still matches, both
    of them exactly as ripgrep has it. An absolute name strips against nothing, and
    the empty anchor is that verdict: it leaves the path absolute, where `/x.txt` and
    `sub/y.txt` match nothing and a bare `*.swp` still matches on the name alone.

    Read once per walk rather than per directory, so that one answer covers all of it,
    nested checkouts included.
    """
    if spec is None:
        return ()
    working_directory = _dir_prefix(Path.cwd())
    # The walk spells every path it finds as `named.absolute()` with what it found
    # hung off that, so a relative name always strips: `..` becomes `../x.txt` here,
    # which is the spelling ripgrep matches too. Normalising the name first instead
    # left a `..` root anchored at nothing, and a rule reaching through the `..`
    # then bit for ripgrep and not for toko.
    anchor = "" if named.is_absolute() else working_directory
    return (_IgnoreLayer(_Anchor(anchor), spec),)


class _WalkOptions(NamedTuple):
    base_prefix: str
    recursive: bool
    respect_gitignore: bool
    respect_dot_ignore: bool
    include_hidden: bool
    follow_symlinks: bool
    exclude_spec: pathspec.PathSpec | None
    global_layers: tuple[_IgnoreLayer, ...]
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


def _excluded(spec: pathspec.PathSpec, relative: str, *, is_dir: bool) -> bool:
    if not is_dir:
        return spec.match_file(relative)
    # Patterns are walked in order because the last match is the one to win, and each
    # is shown the spelling _probe picks for it rather than one spelling for all.
    excluded = False
    for pattern in spec.patterns:
        if pattern.include is None:
            continue
        if pattern.match_file(_probe(relative, pattern, is_dir=True)) is not None:
            excluded = pattern.include
    return excluded


def _should_skip(
    absolute_path: str, frame: _Frame, options: _WalkOptions, *, is_dir: bool = False
) -> bool:
    if frame.rules.matches(absolute_path, is_dir=is_dir):
        return True
    if options.exclude_spec is None:
        return False
    relative = absolute_path.removeprefix(options.base_prefix)
    return _excluded(options.exclude_spec, relative, is_dir=is_dir)


def _describe(path: str, error: OSError) -> str:
    return f"{path}: {error.strerror or error}"


def _lead_from(directory: Path, base_root: Path) -> str:
    """Name the walk root relative to a directory at or above it, ready to prepend."""
    relative = base_root.relative_to(directory)
    return "" if relative == Path() else f"{relative}{os.sep}"


def _initial_frame(base_dir: Path, options: _WalkOptions) -> _Frame:
    """Build the base frame: every ignore file above the base that still reaches it.

    Which directories lie above the walk root is a question about the directory, not
    about the name it was reached by, so it is asked of the root resolved. The spelling
    is still what the excludes file is matched against, and `options.base_prefix` still
    carries it -- these are two uses of one path with opposite needs, and answering
    either from the other's quantity is a bug in one direction or the other.
    """
    # `Path.absolute` leaves `..` where it is, so `/w/sub/..`.parents began at `/w/sub`
    # -- the working directory, a *child* of the root being walked -- and every ignore
    # file in it was installed as a parent layer governing the whole walk. `resolve`
    # rather than `os.path.abspath` because a `..` reached through a symlink lands
    # where the link points and not beside it, which is the directory ripgrep walks.
    base_root = base_dir.resolve()
    repo_root = _find_repo_root(base_root) if options.respect_gitignore else None
    git: list[_IgnoreLayer] = []
    ignore: list[_IgnoreLayer] = []
    rgignore: list[_IgnoreLayer] = []
    if options.respect_gitignore:
        if repo_root is not None:
            git.extend(options.global_layers)
        for directory in (*reversed(base_root.parents), base_root):
            anchor = _Anchor(options.base_prefix, _lead_from(directory, base_root))
            if _is_within(directory, repo_root):
                git.extend(_git_layers(directory, _git_dir(directory), anchor))
            # .ignore and .rgignore are ripgrep's own, not git's, so a repository
            # boundary does not stop them the way it stops the git files.
            if options.respect_dot_ignore:
                ignore.extend(_dot_layer(directory, ".ignore", anchor))
                rgignore.extend(_dot_layer(directory, ".rgignore", anchor))
    ancestors: tuple[tuple[tuple[int, int], Path], ...] = ()
    if options.follow_symlinks:
        try:
            info = base_dir.stat()
        except OSError:
            info = None
        if info is not None:
            ancestors = (((info.st_dev, info.st_ino), base_dir),)
    return _Frame(
        base_dir,
        options.base_prefix,
        _IgnoreRules(tuple(git), tuple(ignore), tuple(rgignore)),
        repo_root,
        ancestors,
    )


def _descend(parent: _Frame, name: str, options: _WalkOptions) -> _Frame:
    """Build a subdirectory's frame, stacking its own ignore files onto its parent's."""
    prefix = parent.prefix + name + os.sep
    directory = Path(prefix.rstrip(os.sep))
    repo_root = parent.repo_root
    git = parent.rules.git
    ignore = parent.rules.ignore
    rgignore = parent.rules.rgignore
    if options.respect_gitignore:
        anchor = _Anchor(prefix)
        git_dir = _git_dir(directory)
        if git_dir is not None:
            # A checkout nested inside another one answers to its own git ignore files
            # and no further: git resolves a path against the repository it belongs to,
            # and ripgrep follows it, so the outer rules are dropped rather than kept.
            repo_root = directory
            git = options.global_layers
        if repo_root is not None:
            git = (*git, *_git_layers(directory, git_dir, anchor))
        if options.respect_dot_ignore:
            ignore = (*ignore, *_dot_layer(directory, ".ignore", anchor))
            rgignore = (*rgignore, *_dot_layer(directory, ".rgignore", anchor))
    rules = _IgnoreRules(git, ignore, rgignore)
    return _Frame(parent.path / name, prefix, rules, repo_root)


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
    stack = [_initial_frame(base_dir, options)]
    while stack:
        frame = stack.pop()
        try:
            # `with`, because an OSError raised part-way through the iteration leaves
            # the iterator unexhausted, and an unexhausted ScandirIterator holds an
            # open directory handle until the garbage collector gets to it.
            with os.scandir(frame.path) as scan:
                entries = list(scan)
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
                # Ahead of both the recursion and the ignore checks, because ripgrep
                # reports a loop it has no intention of descending: through a
                # directory an ignore rule prunes, and at a depth it stops at. Judging
                # those first left the run silent and exiting 0 on a tree rg calls
                # broken. _inspect has already paid for `ident`, so this costs no stat.
                ancestor = _looped_ancestor(frame, found.ident)
                if ancestor is not None:
                    options.report(
                        f"File system loop found: {entry.path} points to an "
                        f"ancestor {ancestor}"
                    )
                    continue
                if not options.recursive or _should_skip(
                    absolute, frame, options, is_dir=True
                ):
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

    Which ignore files apply is a question about the directory `path` names and not
    about the name: `Path("sub")`, `Path.cwd() / "sub"` and `Path("../here/sub")` all
    answer to one set of `.gitignore`, `.ignore` and `.rgignore` files, the ones at and
    above that directory. The earlier claim here, that the three could come back with
    three different listings, described a bug rather than a decision -- a walk root
    spelled through `..` reported the working directory among its own ancestors and was
    governed by ignore files sitting *below* it.

    The spelling does still move one answer, and only one: `core.excludesFile`, which
    ripgrep anchors at the working directory and matches each path against as the caller
    wrote it. A rule of `*/x.txt` there drops `sub/x.txt` and leaves the same file named
    absolutely alone. toko is held to ripgrep's answer on both counts, and holding it
    there means departing from git for an ignore file *above* the walk root: ripgrep
    puts each entry to those rules under the walk root joined with the entry's own
    name, so walking `root`, `/root/*.bb` in the parent's ignore file drops
    `root/sub/deep.bb` while `/root/sub/deep.bb` there leaves it alone. See `_Anchor`.

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
        global_layers=(
            _global_layers(_global_excludes_spec(), path) if respect_gitignore else ()
        ),
        report=on_error if on_error is not None else lambda _message: None,
    )
    return sorted(_walk(path, options))


def read_file(path: Path) -> str:
    """Read a file as UTF-8 text.

    The encoding is named rather than left to the locale. Every caller here reads a
    UnicodeDecodeError as "this file is binary", so a default that is not UTF-8 would
    quietly drop every UTF-8 file in the tree from the count instead of counting it.
    """
    return path.read_text(encoding="utf-8")


def fetch_url(url: str, *, timeout: float = 30.0) -> str:
    """Fetch the body of a URL as text.

    httpx decodes with errors="replace", so a body that is not valid text comes back
    with replacement characters instead of raising.
    """
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text
