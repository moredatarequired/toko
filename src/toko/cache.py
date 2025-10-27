"""Token count caching using SQLite."""

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

_CACHE_DIR_OVERRIDE: Path | None = None


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


def get_cache_db_path() -> Path:
    if _CACHE_DIR_OVERRIDE is not None:
        cache_dir = _CACHE_DIR_OVERRIDE
    else:
        cache_dir = _default_cache_root() / "toko"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "token_cache.db"


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
        with sqlite3.connect(cache_path) as conn:
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
        with sqlite3.connect(cache_path) as conn:
            _init_db(conn)

            # Get existing counts for this message
            cursor = conn.execute(
                "SELECT counts_json FROM token_counts WHERE message_hash = ?",
                (message_hash,),
            )
            row = cursor.fetchone()

            if row:
                # Update existing entry
                counts = json.loads(row[0])
                counts[model] = count
                conn.execute(
                    "UPDATE token_counts SET counts_json = ? WHERE message_hash = ?",
                    (json.dumps(counts), message_hash),
                )
            else:
                # Insert new entry
                counts = {model: count}
                conn.execute(
                    "INSERT INTO token_counts (message_hash, counts_json) VALUES (?, ?)",
                    (message_hash, json.dumps(counts)),
                )

            conn.commit()
    except (sqlite3.Error, json.JSONDecodeError):
        # Silently fail - caching is optional
        pass


def clear_cache() -> None:
    cache_path = get_cache_db_path()
    if cache_path.exists():
        cache_path.unlink()
