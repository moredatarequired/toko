"""One reused httpx client per process, rather than one per request."""

import atexit
import http.cookiejar
import os
import threading
from urllib.request import getproxies_environment

import httpx

# Two pieces of environment are read once, in the constructor, and never revisited: the
# proxies httpx mounts transports for, and the certificates create_ssl_context loads
# from SSL_CERT_FILE or SSL_CERT_DIR. A single process-wide client would freeze both at
# whatever they were when the first request went out, so the client is keyed on exactly
# those two and a change to either retires it. The rest of what the constructor settles
# is our own configuration, which does not vary within a run.
#
# httpx mounts proxies for these four schemes only, ignoring every other *_proxy
# variable in the environment (TRAVIS_APT_PROXY and friends).
_PROXY_SCHEMES = ("all", "http", "https", "no")
# httpx reads the certificate variables in this order and uses only the first one set.
_CERTIFICATE_VARIABLES = ("SSL_CERT_FILE", "SSL_CERT_DIR")

_CLIENTS: dict[tuple[tuple[str, str], ...], httpx.Client] = {}
_LOCK = threading.Lock()


class _DiscardingCookieJar(http.cookiejar.CookieJar):
    """Forget every Set-Cookie rather than replaying it on a later request.

    A client per request started with an empty jar every time; one client for the whole
    run would otherwise send a cookie picked up fetching one URL along with a fetch of
    another the user asked for separately. No call site here wants a session.
    """

    def extract_cookies(self, response: object, request: object) -> None:
        pass


def _client_environment() -> tuple[tuple[str, str], ...]:
    # Read the proxy environment the way httpx does, through urllib, so that the key
    # does not split on spellings urllib folds together, values it discards as empty, or
    # variables httpx never mounts. (httpx calls getproxies(), which on macOS also
    # consults system configuration; that is not environment, and asking for it on every
    # request would be a syscall in a hot path.)
    proxies = getproxies_environment()
    certificates = [
        (name, os.environ[name])
        for name in _CERTIFICATE_VARIABLES
        if os.environ.get(name)
    ]
    return (
        *((scheme, proxies[scheme]) for scheme in _PROXY_SCHEMES if scheme in proxies),
        *certificates[:1],
    )


def shared_client() -> httpx.Client:
    key = _client_environment()
    with _LOCK:
        client = _CLIENTS.get(key)
        # A client someone else has closed is unusable: every request through it raises
        # RuntimeError, which is neither an httpx.HTTPError nor a ValueError and so
        # escapes every handler above. Replace it rather than hand it out.
        if client is not None and not client.is_closed:
            return client
        # Nothing needs two live clients at once, so a new key retires the old client
        # instead of stacking another alongside it and leaking its keepalive sockets.
        superseded = list(_CLIENTS.values())
        _CLIENTS.clear()
        client = _CLIENTS[key] = httpx.Client(cookies=_DiscardingCookieJar())
    # Closing outside the lock: close() waits on the connection pool, and nothing that
    # holds a pooled connection should be waiting on this lock behind it.
    for stale in superseded:
        stale.close()
    return client


@atexit.register
def close_shared_clients() -> None:
    with _LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        client.close()
