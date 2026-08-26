"""One httpx client is reused across a run rather than rebuilt for every request."""

import gc
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import toko.counter as counter
from toko.cli import app
from toko.counter import count_tokens
from toko.file_reader import fetch_url
from toko.http_client import close_shared_clients, shared_client

runner = CliRunner()

MODEL = "claude-opus-4-5"


def _set_https_proxy(monkeypatch, url: str) -> None:
    """Set both spellings: urllib prefers the lowercase one, whatever the run inherited."""
    monkeypatch.setenv("HTTPS_PROXY", url)
    monkeypatch.setenv("https_proxy", url)


@pytest.fixture(autouse=True)
def _fresh_clients():
    """Start and end every test with an empty cache; it is process-global."""
    close_shared_clients()
    try:
        yield
    finally:
        close_shared_clients()


@pytest.fixture
def local_api(monkeypatch):
    """Stand in for the Anthropic count endpoint with a keep-alive HTTP server.

    It counts connections as well as requests, which is what tells reuse from rebuild:
    a client per request cannot keep a connection alive across two of them.
    """
    lock = threading.Lock()
    tally = SimpleNamespace(requests=0, connections=0, bodies=[], cookies=[])

    class Handler(BaseHTTPRequestHandler):
        # Without keep-alive the server closes every connection itself and the
        # connection count below would be one per request whatever the client does.
        protocol_version = "HTTP/1.1"
        # BaseHTTPRequestHandler writes headers and body separately, which on a
        # kept-alive connection meets delayed ACK and stalls each request by 40ms.
        disable_nagle_algorithm = True

        def setup(self):
            super().setup()
            with lock:
                tally.connections += 1

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            with lock:
                tally.requests += 1
                tally.bodies.append(body)
            text = json.loads(body)["messages"][0]["content"]
            payload = json.dumps({"input_tokens": len(text)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            with lock:
                tally.requests += 1
                tally.cookies.append(self.headers.get("Cookie"))
            payload = b"fetched over the shared client"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            # Plenty of sites set one; a client that keeps it sends it back.
            self.send_header("Set-Cookie", "sid=abc123; Path=/")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            """Keep the server quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # httpx honours HTTP_PROXY for http:// URLs without exempting loopback, so without
    # this every request here goes to the developer's proxy. Both spellings, because
    # urllib reads them both and prefers the lowercase one.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(counter, "ANTHROPIC_COUNT_URL", f"{base}/v1/messages")

    try:
        yield SimpleNamespace(base=base, tally=tally)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_every_caller_gets_the_same_client():
    assert shared_client() is shared_client()


def test_every_thread_gets_the_same_client():
    with ThreadPoolExecutor(max_workers=8) as pool:
        clients = list(pool.map(lambda _: shared_client(), range(64)))

    assert len(set(map(id, clients))) == 1


def test_a_changed_proxy_environment_gets_its_own_client(monkeypatch):
    """Httpx mounts proxy transports in the constructor and never revisits them.

    Reusing one client regardless would pin a run to whatever proxies happened to be
    set when its first request went out.
    """
    _set_https_proxy(monkeypatch, "http://proxy.invalid:3128")
    first = shared_client()
    _set_https_proxy(monkeypatch, "http://other-proxy.invalid:3128")

    assert shared_client() is not first


def test_counting_many_files_holds_one_connection_open(local_api):
    texts = [f"body number {index}" for index in range(40)]

    for text in texts:
        count_tokens(text, model=MODEL, use_cache=False)

    assert local_api.tally.requests == 40
    # One connection is what this sees; the bound is three times that for the reason
    # given below, and still nowhere near the 40 that a client per request produces.
    assert local_api.tally.connections <= 3


def test_connections_are_bounded_by_jobs_rather_than_by_file_count(local_api):
    """The peak that crashed a wide run scaled with concurrency, not with the count."""

    def count(index: int) -> int:
        return count_tokens(f"body {index}", model=MODEL, use_cache=False).count

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(count, range(200)))

    assert local_api.tally.requests == 200
    assert len(counts) == 200
    # Eight workers need eight connections, and eight is what this sees. The bound
    # is three times that because a connection can be retired and replaced mid-run and
    # a tight one would flake in CI; it is still far below the 200 that one client per
    # request produces.
    assert local_api.tally.connections <= 24


@pytest.mark.usefixtures("local_api")
def test_a_concurrent_run_counts_the_same_as_a_sequential_one(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(24):
        (tree / f"f{index:02d}.txt").write_text("x" * (index + 1))
    args = ["--header", "--format", "csv", "-m", MODEL, str(tree)]

    # Concurrent first: counts are cached, so a sequential first pass would leave the
    # concurrent one replaying the cache instead of counting anything.
    concurrent = runner.invoke(app, [*args, "--jobs", "8"])
    sequential = runner.invoke(app, [*args, "--jobs", "1"])

    assert concurrent.exit_code == 0
    assert sequential.exit_code == 0
    assert concurrent.stdout == sequential.stdout
    # Not vacuous: every file counts differently, so a count landing on the wrong row
    # would change the output.
    counts = [line.split(",")[-1] for line in concurrent.stdout.splitlines()[1:]]
    assert counts == [str(index + 1) for index in range(24)]


def test_fetch_url_goes_through_the_shared_client(local_api):
    client = shared_client()

    for index in range(8):
        fetched = fetch_url(f"{local_api.base}/{index}.txt")
        assert fetched == "fetched over the shared client"

    assert local_api.tally.requests == 8
    assert local_api.tally.connections <= 3
    assert shared_client() is client


@pytest.mark.usefixtures("local_api")
def test_closing_the_shared_clients_leaves_no_socket_behind():
    count_tokens("hello", model=MODEL, use_cache=False)
    gc.collect()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        close_shared_clients()
        gc.collect()

    leaked = [
        warning for warning in recorded if issubclass(warning.category, ResourceWarning)
    ]
    assert leaked == []


@pytest.mark.usefixtures("local_api")
def test_a_closed_client_is_replaced_on_the_next_call():
    first = shared_client()
    close_shared_clients()

    assert first.is_closed
    second = shared_client()
    assert second is not first
    assert count_tokens("hello", model=MODEL, use_cache=False).count == 5


def test_proxy_variables_httpx_never_reads_do_not_split_the_cache(monkeypatch):
    """A client per spelling is a client per keepalive pool, for no change in behaviour."""
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    first = shared_client()

    # An unrelated *_proxy variable, which httpx deliberately does not mount.
    monkeypatch.setenv("TRAVIS_APT_PROXY", "http://apt.invalid:3128")
    # An empty value, which urllib discards rather than reading as a proxy.
    monkeypatch.setenv("ALL_PROXY", "")
    # A second spelling of one already set, which loses to the lowercase one.
    monkeypatch.setenv("HTTP_PROXY", "http://never-read.invalid:3128")

    assert shared_client() is first


def test_a_changed_certificate_environment_gets_its_own_client(monkeypatch, tmp_path):
    """create_ssl_context reads the CA bundle in the constructor, as it does the proxies.

    A process that points at a new bundle after its first request would otherwise keep
    verifying against the old one for the rest of its life.
    """
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    first = shared_client()
    monkeypatch.setenv("SSL_CERT_DIR", str(other))

    assert shared_client() is not first


def test_a_superseded_client_is_closed_rather_than_left_holding_its_sockets(
    monkeypatch,
):
    """Environment churn used to stack up clients, each with its own keepalive pool."""
    clients = []
    for index in range(5):
        _set_https_proxy(monkeypatch, f"http://proxy-{index}.invalid:3128")
        clients.append(shared_client())

    assert len({id(client) for client in clients}) == 5
    assert [client.is_closed for client in clients] == [True, True, True, True, False]


@pytest.mark.usefixtures("local_api")
def test_a_client_closed_by_its_caller_is_replaced_rather_than_handed_out_again():
    """`with shared_client() as client:` closes the client the whole process shares.

    Handing that one back would raise RuntimeError on every later request, which is
    neither an httpx.HTTPError nor a ValueError and so escapes every handler above.
    """
    first = shared_client()
    first.close()

    assert shared_client() is not first
    assert count_tokens("hello", model=MODEL, use_cache=False).count == 5


def test_a_cookie_from_one_url_is_not_sent_to_the_next(local_api):
    """One client for the run must not turn separate fetches into a session."""
    for index in range(3):
        assert fetch_url(f"{local_api.base}/{index}.txt")

    assert local_api.tally.cookies == [None, None, None]
