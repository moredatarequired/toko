"""Token count caching using SQLite."""

import hashlib
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

_CACHE_DIR_OVERRIDE: Path | None = None

# How long to wait for another writer to release the database. Well above sqlite3's
# 5-second default because the writers are not just this run's threads: several toko
# processes counting at once all serialise on the one file, and a wait that expires is
# a dropped cache entry, since the failure is deliberately swallowed below.
_BUSY_TIMEOUT_SECONDS = 30.0


def _default_cache_root() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base)
        return Path.home() / "AppData" / "Local"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"

    return Path.home() / ".cache"


def get_cache_dir() -> Path:
    if _CACHE_DIR_OVERRIDE is not None:
        cache_dir = _CACHE_DIR_OVERRIDE
    else:
        cache_dir = _default_cache_root() / "toko"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_db_path() -> Path:
    return get_cache_dir() / "token_cache.db"


def set_cache_dir(path: str | Path | None) -> None:
    """Override the cache directory used for the SQLite database."""
    if path is None:
        globals()["_CACHE_DIR_OVERRIDE"] = None
        return
    cache_path = Path(path)
    globals()["_CACHE_DIR_OVERRIDE"] = cache_path


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_counts (
            message_hash TEXT PRIMARY KEY,
            counts_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _hash_message(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get_cached_count(text: str, model: str) -> int | None:
    cache_path = get_cache_db_path()
    if not cache_path.exists():
        return None

    message_hash = _hash_message(text)

    try:
        # closing() around connect(), not just `with connect(...)`: the connection's own
        # context manager commits or rolls back but never closes the handle, and a
        # sqlite3.Connection sits in a reference cycle, so only the cyclic collector
        # reclaims it. That collector does not run in pool worker threads, so without
        # this the descriptors grow one per counted file until the process runs out. The
        # nested `with conn:` is kept as belt and braces but does nothing here: a bare
        # SELECT opens no transaction for it to end.
        with (
            closing(sqlite3.connect(cache_path, timeout=_BUSY_TIMEOUT_SECONDS)) as conn,
            conn,
        ):
            cursor = conn.execute(
                "SELECT counts_json FROM token_counts WHERE message_hash = ?",
                (message_hash,),
            )
            row = cursor.fetchone()

            if row:
                counts = json.loads(row[0])
                return counts.get(model)
    except (sqlite3.Error, json.JSONDecodeError):
        return None

    return None


def cache_count(text: str, model: str, count: int) -> None:
    cache_path = get_cache_db_path()
    message_hash = _hash_message(text)

    try:
        # closing() for the same reason as above. The nested `with conn:` is not what
        # makes the upsert atomic either: it issues nothing on entry, the INSERT's own
        # implicit BEGIN and the explicit commit() below are the transaction, and it is
        # SQLite's file locking that holds off other processes.
        with (
            closing(sqlite3.connect(cache_path, timeout=_BUSY_TIMEOUT_SECONDS)) as conn,
            conn,
        ):
            _init_db(conn)

            # Merged in SQL rather than read here and written back, because counts run
            # concurrently: two models counted against the same text would both see no
            # row, both insert, and the loser's count would be lost to the UNIQUE
            # constraint and swallowed below. json_patch folds the new model into
            # whatever is stored, so a writer need not have read the others' work.
            conn.execute(
                """
                INSERT INTO token_counts (message_hash, counts_json) VALUES (?, ?)
                ON CONFLICT(message_hash) DO UPDATE
                    SET counts_json = json_patch(counts_json, excluded.counts_json)
                """,
                (message_hash, json.dumps({model: count})),
            )

            conn.commit()
    except sqlite3.Error:
        # Silently fail - caching is optional
        pass


def clear_cache() -> None:
    cache_path = get_cache_db_path()
    if cache_path.exists():
        cache_path.unlink()
