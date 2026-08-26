"""One reused httpx client per process, rather than one per request."""

import atexit
import os
import threading

import httpx

# httpx reads the proxy environment once, in the constructor, and mounts transports for
# what it finds there. A single process-wide client would therefore freeze whatever
# proxies happened to be set when the first request went out, so the cache is keyed on
# that environment and a change to it gets its own client.
_CLIENTS: dict[tuple[tuple[str, str], ...], httpx.Client] = {}
_LOCK = threading.Lock()


def _proxy_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.lower().endswith("_proxy")
        )
    )


def shared_client() -> httpx.Client:
    with _LOCK:
        key = _proxy_environment()
        client = _CLIENTS.get(key)
        if client is None:
            client = _CLIENTS[key] = httpx.Client()
        return client


@atexit.register
def close_shared_clients() -> None:
    with _LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        client.close()
