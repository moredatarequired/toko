"""API keys must never reach a user-visible error message, warning or traceback."""

import inspect
import json
import os
import select
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
import respx
from typer.testing import CliRunner

import toko.counter as counter
from toko.cli import app
from toko.counter import (
    GOOGLE_COUNT_URL_BASE,
    _describe_request_failure,
    _redact_key,
    count_tokens,
)

try:
    import fcntl
    import pty
    import termios

    HAS_PTY = True
except ImportError:  # pty, termios and fcntl are POSIX-only
    HAS_PTY = False

runner = CliRunner()

SENTINEL = "toko-test-sentinel-do-not-log"

# os.environ hands back a byte that is not valid UTF-8 as a surrogate escape, and a
# surrogate is exactly what UTF-8 cannot encode.
NON_UTF8_KEY_BODY = "toko-key-with-a-raw-byte-do-not-log"
NON_UTF8_KEY = f"{NON_UTF8_KEY_BODY}\udce9"

# (environment variable, model, phrase the failure message opens with)
PROVIDERS = [
    pytest.param("GOOGLE_API_KEY", "gemini-2.5-flash", "Google", id="google"),
    pytest.param("ANTHROPIC_API_KEY", "claude-sonnet-4-5", "Anthropic", id="anthropic"),
    pytest.param("XAI_API_KEY", "grok-4.5", "xAI", id="xai"),
]

# (environment variable, model, a body it counts, the header carrying the key and the
# value that header must hold once the key has been stripped)
KEY_HEADER_PROVIDERS = [
    pytest.param(
        "GOOGLE_API_KEY",
        "gemini-2.5-flash",
        {"totalTokens": 4},
        "x-goog-api-key",
        SENTINEL,
        id="google",
    ),
    pytest.param(
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-5",
        {"input_tokens": 4},
        "x-api-key",
        SENTINEL,
        id="anthropic",
    ),
    pytest.param(
        "XAI_API_KEY",
        "grok-4.5",
        {"token_count": 4},
        "Authorization",
        f"Bearer {SENTINEL}",
        id="xai",
    ),
]


@pytest.fixture
def local_api(monkeypatch):
    """Point every provider at a real HTTP server on localhost.

    respx intercepts above the transport, but httpx rejects an illegal header value
    while writing the request -- below where respx sits -- so a mocked route cannot
    reach the code path that used to echo the key back.
    """
    requests = []
    status = [200]
    body: list[object] = [{}]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            requests.append(dict(self.headers))
            payload = json.dumps(body[0]).encode()
            self.send_response(status[0])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            """Keep the server quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # httpx honours HTTP_PROXY for http:// URLs and does not exempt loopback on its
    # own, so without this every request here is posted to the developer's proxy.
    # Both spellings, because whichever appears later in the environment wins.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(counter, "GOOGLE_COUNT_URL_BASE", base)
    monkeypatch.setattr(counter, "ANTHROPIC_COUNT_URL", f"{base}/v1/messages")
    monkeypatch.setattr(counter, "XAI_TOKENIZE_URL", f"{base}/v1/tokenize")
    # The xAI counter falls back to a local tokenizer, which would hide API failures.
    monkeypatch.setattr(counter, "HAS_TRANSFORMERS", False)

    def respond(new_status: int, new_body: object) -> None:
        status[0] = new_status
        body[0] = new_body

    try:
        yield SimpleNamespace(respond=respond, requests=requests, base=base)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "control", ["\n", "\r", "\x0b"], ids=["newline", "carriage-return", "vertical-tab"]
)
@pytest.mark.parametrize(("env_var", "model", "provider"), PROVIDERS)
def test_a_key_holding_a_control_character_does_not_leak(
    local_api, monkeypatch, env_var, model, provider, control
):
    """Httpx refuses to send such a header and quotes the raw bytes back at us.

    Stripping only helps at the ends, and replacing the key in that text cannot work:
    the message holds a two-character escape sequence where the key holds one byte.
    """
    head, tail = "toko-key-head-do-not-log", "toko-key-tail-do-not-log"
    monkeypatch.setenv(env_var, f"{head}{control}{tail}")
    local_api.respond(401, {"error": "unauthorized"})

    with pytest.raises(ValueError, match=f"tokens for {provider}") as excinfo:
        count_tokens("hello", model=model, use_cache=False)

    message = str(excinfo.value)
    assert head not in message
    assert tail not in message
    assert "Illegal header value" not in message
    # A user still has to be able to act on this.
    assert "LocalProtocolError" in message
    assert local_api.base in message


@pytest.mark.parametrize(
    ("env_var", "model", "body", "header", "sent"), KEY_HEADER_PROVIDERS
)
def test_a_trailing_newline_is_stripped_so_the_request_still_goes_out(
    local_api, monkeypatch, env_var, model, body, header, sent
):
    """`export ANTHROPIC_API_KEY=$(cat keyfile)` leaves one behind.

    Every provider resolves its key through the same helper, so every provider has to
    survive the same environment.
    """
    monkeypatch.setenv(env_var, f"{SENTINEL}\n")
    local_api.respond(200, body)

    assert count_tokens("hello", model=model, use_cache=False).count == 4
    assert local_api.requests[-1][header] == sent


@pytest.mark.parametrize(("env_var", "model", "provider"), PROVIDERS)
def test_a_response_body_that_echoes_the_key_is_redacted(
    local_api, monkeypatch, env_var, model, provider
):
    """The one place a key can still reach the message text is the provider's own body."""
    monkeypatch.setenv(env_var, SENTINEL)
    local_api.respond(200, {"detail": f"key {SENTINEL} is not authorized"})

    with pytest.raises(ValueError, match="Unexpected response") as excinfo:
        count_tokens("hello", model=model, use_cache=False)

    message = str(excinfo.value)
    assert SENTINEL not in message
    assert "***" in message
    assert provider in message


def test_an_echoed_key_is_redacted_from_the_xai_approximation_warning(
    local_api, monkeypatch, capsys
):
    monkeypatch.setenv("XAI_API_KEY", SENTINEL)
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 7)
    local_api.respond(200, {"detail": f"key {SENTINEL} is not authorized"})

    assert count_tokens("hello", model="grok-4.5", use_cache=False).count == 7

    stderr = capsys.readouterr().err
    assert SENTINEL not in stderr
    assert "***" in stderr


def test_an_echoed_key_is_redacted_from_the_caveat_json_prints_on_stdout(
    local_api, monkeypatch, tmp_path
):
    """The warning and `TokenCount.caveat` are one string, and JSON puts it on stdout.

    Redacting only on the way to stderr would leave the same text leaking through a
    piped, machine-read document -- the likelier place for it to be captured.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", SENTINEL)
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 7)
    local_api.respond(200, {"detail": f"key {SENTINEL} is not authorized"})

    result = runner.invoke(app, ["--format", "json", "-m", "grok-4.5", "-t", "hello"])

    assert result.exit_code == 0
    assert SENTINEL not in result.stdout
    # Not vacuous: the caveat is on stdout, it is just redacted.
    entry = json.loads(result.stdout)["grok-4.5"]
    assert entry["tokens"] == 7
    assert "***" in entry["caveat"]


def test_the_xai_approximation_survives_a_key_holding_a_non_utf8_byte(
    local_api, monkeypatch, capsys
):
    """Redacting such a key used to raise, taking the fallback count down with it.

    The user saw "All models failed to count tokens" instead of an approximate count.
    """
    monkeypatch.setenv("XAI_API_KEY", NON_UTF8_KEY)
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 7)

    assert count_tokens("hello", model="grok-4.5", use_cache=False).count == 7

    stderr = capsys.readouterr().err
    assert "approximate" in stderr
    assert NON_UTF8_KEY_BODY not in stderr
    # httpx refuses such a header before sending, so the server never sees a request.
    assert local_api.base in stderr
    assert local_api.requests == []


@pytest.mark.parametrize(("env_var", "model", "provider"), PROVIDERS)
def test_a_json_list_body_raises_value_error_rather_than_tracebacking(
    local_api, monkeypatch, env_var, model, provider
):
    """A 200 whose body is a list used to raise AttributeError from data.get().

    AttributeError is not a ValueError, so it escaped the CLI's handler and reached
    Typer's traceback renderer -- with the key sitting in the frame locals.
    """
    monkeypatch.setenv(env_var, SENTINEL)
    local_api.respond(200, [{"input_tokens": 4, "totalTokens": 4, "token_count": 4}])

    with pytest.raises(ValueError, match="Unexpected response") as excinfo:
        count_tokens("hello", model=model, use_cache=False)

    assert provider in str(excinfo.value)


@respx.mock
@pytest.mark.parametrize("status", [401, 404])
def test_google_http_error_reports_the_status_without_the_key(monkeypatch, status):
    monkeypatch.setenv("GOOGLE_API_KEY", SENTINEL)
    respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(status, json={"error": "nope"})
    )

    with pytest.raises(
        ValueError, match="Failed to count tokens for Google"
    ) as excinfo:
        count_tokens("hello", model="gemini-2.5-flash", use_cache=False)

    message = str(excinfo.value)
    assert SENTINEL not in message
    assert f"HTTP {status}" in message


@respx.mock
def test_google_key_travels_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", SENTINEL)
    route = respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(200, json={"totalTokens": 3})
    )

    assert count_tokens("hello", model="gemini-2.5-flash", use_cache=False).count == 3

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == SENTINEL
    assert SENTINEL not in str(request.url)


# Rich truncates a string in a frame-locals panel to 80 characters, so keep the
# sentinel under 80: a longer one could not be found in the rendered traceback even
# when it did leak. (A real Anthropic key is longer than that, and would have leaked
# as its first 80 characters -- still a total compromise.)
TRACEBACK_SENTINEL = "LEAKYKEY42"

_TRACEBACK_DRIVER = """
import os

from toko.cli import app


@app.command("raise-for-test")
def raise_for_test() -> None:
    api_key = os.environ["TOKO_TRACEBACK_SENTINEL"]
    headers = {"x-api-key": api_key}
    raise RuntimeError(f"deliberate failure carrying {len(headers)} header")


app(["raise-for-test"])
"""


_PTY_READ_TIMEOUT = 60.0


def _run_under_pty(script: str, env: dict[str, str]) -> str:
    """Run a script attached to a terminal, since Rich renders differently without one."""
    primary, secondary = pty.openpty()
    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, script],
        stdin=subprocess.DEVNULL,
        stdout=secondary,
        stderr=secondary,
        env=env,
    )
    os.close(secondary)
    chunks = []
    deadline = time.monotonic() + _PTY_READ_TIMEOUT
    timed_out = False
    try:
        while True:
            # A child hanging with the pty still open never reaches EIO, and an untimed
            # read would then hang the whole suite instead of failing this one test.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if not select.select([primary], [], [], remaining)[0]:
                continue
            try:
                data = os.read(primary, 65536)
            except OSError:  # the terminal reports EIO once the child is gone
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(primary)
        if timed_out:
            process.kill()
        process.wait(timeout=10)
    if timed_out:
        pytest.fail(f"child did not exit within {_PTY_READ_TIMEOUT}s")
    return b"".join(chunks).decode(errors="replace")


@pytest.mark.skipif(not HAS_PTY, reason="pty, termios and fcntl are POSIX-only")
def test_a_rendered_traceback_does_not_show_frame_locals(tmp_path):
    """An unhandled error must not print the api_key held by the frame that raised."""
    script = tmp_path / "traceback_driver.py"
    script.write_text(_TRACEBACK_DRIVER)
    env = dict(
        os.environ,
        TOKO_TRACEBACK_SENTINEL=TRACEBACK_SENTINEL,
        COLUMNS="200",
        TERM="xterm-256color",
    )

    output = _run_under_pty(str(script), env)

    # Without this the leak assertion below would pass for want of a traceback.
    assert "RuntimeError" in output
    assert "deliberate failure" in output
    assert TRACEBACK_SENTINEL not in output


@pytest.mark.parametrize("empty", ["", None])
def test_redact_key_is_a_noop_for_empty_keys(empty):
    message = "Client error for url 'https://example.test/v1?key='"
    assert _redact_key(message, empty) == message


def test_redact_key_replaces_raw_and_encoded_forms():
    key = "abc/def+ghijkl"
    message = f"url 'https://example.test?key={key}' header {key}"
    redacted = _redact_key(message, key)

    assert key not in redacted
    assert redacted.count("***") == 2


def test_redact_key_replaces_the_backslash_escaped_form():
    """A key with a control character reaches a repr as two printable characters."""
    key = f"{SENTINEL}\nsecond-line"
    message = f"Illegal header value b'{SENTINEL}\\nsecond-line'"

    assert SENTINEL not in _redact_key(message, key)


def test_redact_key_replaces_percent_encoded_key():
    key = "abc def ghi"
    assert "abc%20def%20ghi" not in _redact_key("url ?key=abc%20def%20ghi", key)


def test_redact_key_leaves_short_values_alone():
    """Blanking a three-character 'key' would mangle unrelated text and protect nothing."""
    assert _redact_key("https://example.test/abcdef", "abc") == (
        "https://example.test/abcdef"
    )


@pytest.mark.parametrize(
    "key",
    [NON_UTF8_KEY, "\ud800-lone-high-surrogate-key"],
    ids=["surrogate-escaped-byte", "lone-high-surrogate"],
)
def test_redact_key_does_not_raise_on_a_key_utf8_cannot_encode(key):
    """quote() used to raise UnicodeEncodeError here -- with the key in its payload."""
    assert _redact_key(f"body echoed {key} back", key) == "body echoed *** back"


def test_redact_key_replaces_the_percent_encoded_form_of_a_non_utf8_byte():
    assert "%E9" not in _redact_key(
        f"url ?key={quote(NON_UTF8_KEY, safe='', errors='surrogateescape')}",
        NON_UTF8_KEY,
    )


def test_describe_request_failure_takes_the_api_key_without_a_default():
    """A default lets a call site drop the argument and skip redaction silently."""
    api_key = inspect.signature(_describe_request_failure).parameters["api_key"]

    assert api_key.default is inspect.Parameter.empty
    assert api_key.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_describe_request_failure_strips_a_query_string_and_fragment():
    """No caller passes a key in the URL today; this pins the guard against that."""
    described = _describe_request_failure(
        httpx.ReadTimeout("slow"),
        f"https://example.test/v1:count?key={SENTINEL}#frag",
        SENTINEL,
    )

    assert described == "ReadTimeout contacting https://example.test/v1:count"


def test_describe_request_failure_does_not_quote_a_protocol_error():
    """A LocalProtocolError's text is the raw header bytes httpx refused to send."""
    described = _describe_request_failure(
        httpx.LocalProtocolError(f"Illegal header value b'Bearer {SENTINEL}'"),
        "https://example.test/v1",
        SENTINEL,
    )

    assert described == "LocalProtocolError contacting https://example.test/v1"


def test_describe_request_failure_keeps_the_errno_behind_a_connect_error():
    """DNS failure and a refused connection otherwise read identically."""
    described = _describe_request_failure(
        httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://example.test/v1",
        SENTINEL,
    )

    assert described == (
        "ConnectError contacting https://example.test/v1: "
        "[Errno -2] Name or service not known"
    )


def test_a_connect_error_is_redacted_even_though_it_cannot_hold_the_key():
    described = _describe_request_failure(
        httpx.ConnectError(f"[Errno 111] {SENTINEL} refused"),
        "https://example.test/v1",
        SENTINEL,
    )

    assert SENTINEL not in described
    assert "[Errno 111] *** refused" in described


def test_a_connect_error_without_a_message_still_says_what_failed():
    described = _describe_request_failure(
        httpx.ConnectError(""), "https://example.test/v1", SENTINEL
    )

    assert described == "ConnectError contacting https://example.test/v1"
