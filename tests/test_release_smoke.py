"""The release smoke test's own checks, run at PR time.

`tests/smoke_test.py` is only invoked by `.github/workflows/release.yml`, against an
installed artifact, after a tag is pushed. Anything that first runs there fails when
publishing is already underway, so everything in it that does not need an installed
artifact is exercised here instead.
"""

import importlib
import itertools
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from tests import smoke_test


def test_optional_groups_still_declare_the_models_the_smoke_test_names():
    """Both names are pinned in the smoke test, so a registry edit must not orphan them."""
    assert smoke_test.MISTRAL_MODEL in smoke_test.optional_models("mistral_common")
    assert smoke_test.TRANSFORMERS_MODEL in smoke_test.optional_models("transformers")

    with pytest.raises(AssertionError, match="No optional group"):
        smoke_test.optional_models("nonesuch")


def test_the_provider_listing_matches_what_the_smoke_test_expects():
    """The exact-set assertion, where adding a provider costs a red PR, not a release."""
    smoke_test.check_listing()


def test_the_mistral_count_is_plausible():
    smoke_test.check_mistral()


# The two environments the guard has to be right in. They are not the same stack: this
# repository is locked to the first, and the release workflow installs the built artifact
# unlocked with `uv run --isolated --no-project --with "dist/toko-*.whl[all]"` and
# resolves the second. Every capture below was induced against the version named here, at
# that literal invocation, on a cold cache.
DEV = "dev(hf-0.36.0/requests, transformers-4.57.3)"
RELEASE = "release(hf-1.27.0/httpx, transformers-5.15.0)"


class Capture(NamedTuple):
    """One induced failure: what the user saw, and what the exception chain carried.

    `message` is the verbatim `str(ValueError)` out of `_count_transformers` and is
    recorded as evidence rather than as input -- the guard never reads it. `chain` is the
    `__cause__`/`__context__` walk from that same failure, outermost first, as
    (fully qualified class, `response.status_code` or None, link kind).

    The link kind is how that link reaches the next one: "cause" for `__cause__`,
    "context" for `__context__`, and None on the innermost link, which reaches nothing.
    It is the edge `_causes` follows, so it has to be recorded rather than assumed --
    the chains are not all `__cause__`, and the places they are not are exactly the
    places that decide a release. See `test_the_guard_must_follow_context_too`.
    """

    message: str
    chain: tuple[tuple[str, int | None, str | None], ...]
    excused: bool

    def replay(self) -> BaseException:
        """Rebuild the recorded chain out of the real classes, link kinds included."""
        links = [_rebuild(dotted, status) for dotted, status, _ in self.chain]
        for index, (outer, inner) in enumerate(itertools.pairwise(links)):
            if self.chain[index][2] == "cause":
                outer.__cause__ = inner
            else:
                outer.__context__ = inner
        return links[0]


def _rebuild(dotted: str, status: int | None) -> BaseException:
    module, _, name = dotted.rpartition(".")
    error_type = getattr(importlib.import_module(module), name)
    # __new__ rather than the constructor: huggingface_hub's exception signatures differ
    # between the two resolved versions, and an unresolvable name should fail this suite
    # loudly rather than quietly drop a case the guard is pinned to.
    error = error_type.__new__(error_type)
    error.args = ("replayed",)
    if status is not None:
        error.response = SimpleNamespace(status_code=status)
    return error


CAPTURES = {
    (DEV, "corrupt-tokenizer"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: Expecting value: line 1 column 1 (char 0)"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("json.decoder.JSONDecodeError", None, "context"),
            ("builtins.StopIteration", None, None),
        ),
        excused=False,
    ),
    (DEV, "get-429"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: There was a specific connection error when trying to load Qwen/Qwen2.5-7B-Instruct:\n"
            "429 Client Error: Too Many Requests for url: http://127.0.0.1:33103/api/resolve-cache/models/Qwen/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28/config.json\n"
            "\n"
            "stand-in injected 429"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("requests.exceptions.HTTPError", 429, None),
        ),
        excused=True,
    ),
    (DEV, "get-503"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: There was a specific connection error when trying to load Qwen/Qwen2.5-7B-Instruct:\n"
            "503 Server Error: Service Unavailable for url: http://127.0.0.1:41313/api/resolve-cache/models/Qwen/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28/config.json"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("requests.exceptions.HTTPError", 503, None),
        ),
        excused=True,
    ),
    (DEV, "head-401"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' requires authentication. Set HF_TOKEN environment variable or run: huggingface-cli login"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 401, "cause"),
            ("requests.exceptions.HTTPError", 401, None),
        ),
        excused=False,
    ),
    (DEV, "head-403-tagged"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' is gated on Hugging Face. Accept the license and provide HF_TOKEN or run: huggingface-cli login"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.GatedRepoError", 403, "cause"),
            ("requests.exceptions.HTTPError", 403, None),
        ),
        excused=False,
    ),
    (DEV, "head-403-untagged"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:55121' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 403, "cause"),
            ("requests.exceptions.HTTPError", 403, None),
        ),
        excused=True,
    ),
    (DEV, "head-404-tagged"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' not found on HuggingFace. Use the full model path (org/model-name). Try: Qwen/Qwen3-8B, Qwen/Qwen2.5-7B"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.RepositoryNotFoundError", 404, "cause"),
            ("requests.exceptions.HTTPError", 404, None),
        ),
        excused=False,
    ),
    (DEV, "head-404-untagged"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:51683' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 404, "cause"),
            ("requests.exceptions.HTTPError", 404, None),
        ),
        excused=True,
    ),
    (DEV, "head-429"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:46561' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("requests.exceptions.HTTPError", 429, None),
        ),
        excused=True,
    ),
    (DEV, "head-500"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:41239' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.HTTPError", 500, None),
        ),
        excused=True,
    ),
    (DEV, "head-503"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:44627' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.HTTPError", 503, None),
        ),
        excused=True,
    ),
    (DEV, "head-connection-refused"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:9' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.ConnectionError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.NewConnectionError", None, "cause"),
            ("builtins.ConnectionRefusedError", None, None),
        ),
        excused=True,
    ),
    (DEV, "head-dns-failure"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://standin-does-not-exist.invalid' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.ConnectionError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.NameResolutionError", None, "cause"),
            ("socket.gaierror", None, None),
        ),
        excused=True,
    ),
    (DEV, "head-timeout"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:44655' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.ReadTimeout", None, "context"),
            ("urllib3.exceptions.ReadTimeoutError", None, "cause"),
            ("builtins.TimeoutError", None, None),
        ),
        excused=True,
    ),
    (DEV, "offline-cold-cache"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, None),
        ),
        excused=True,
    ),
    (DEV, "proxy-answers-cannot-tunnel"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: (MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 502 Bad Gateway')))\"), '(Request ID: 30a38dcc-6fab-4ff0-abee-5fd34f25bafe)')"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.ProxyError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.ProxyError", None, "cause"),
            ("builtins.OSError", None, None),
        ),
        excused=False,
    ),
    (DEV, "proxy-dead"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: (MaxRetryError('HTTPSConnectionPool(host=\\'huggingface.co\\', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json (Caused by ProxyError(\\'Unable to connect to proxy\\', NewConnectionError(\"HTTPSConnection(host=\\'127.0.0.1\\', port=9): Failed to establish a new connection: [Errno 111] Connection refused\")))'), '(Request ID: 9b72d41d-d740-4764-aea5-e60f20f9185f)')"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.ProxyError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.ProxyError", None, "cause"),
            ("urllib3.exceptions.NewConnectionError", None, "cause"),
            ("builtins.ConnectionRefusedError", None, None),
        ),
        excused=False,
    ),
    (DEV, "tls-close-notify"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://127.0.0.1:51849' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("requests.exceptions.ConnectionError", None, "context"),
            ("urllib3.exceptions.ProtocolError", None, "context"),
            ("http.client.RemoteDisconnected", None, None),
        ),
        excused=True,
    ),
    (DEV, "tls-handshake-alert"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: (MaxRetryError(\"HTTPSConnectionPool(host='127.0.0.1', port=56015): Max retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json (Caused by SSLError(SSLError(1, '[SSL: TLSV13_ALERT_CERTIFICATE_REQUIRED] tlsv13 alert certificate required (_ssl.c:2713)')))\"), '(Request ID: a5b878a1-283e-44af-8c4d-3be2c6221878)')"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.SSLError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.SSLError", None, None),
        ),
        excused=False,
    ),
    (DEV, "tls-junk-ca-bundle"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: (MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json (Caused by SSLError(SSLError(136, '[X509: NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found (_ssl.c:4416)')))\"), '(Request ID: 308147bc-bf87-4b91-b880-b27e8c70eada)')"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.SSLError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.SSLError", None, "cause"),
            ("ssl.SSLError", None, None),
        ),
        excused=False,
    ),
    (DEV, "tls-self-signed-endpoint"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: (MaxRetryError(\"HTTPSConnectionPool(host='127.0.0.1', port=34177): Max retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1082)')))\"), '(Request ID: 5de7fd70-382f-4796-8ee6-4addc2b7aff4)')"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.SSLError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.SSLError", None, "context"),
            ("ssl.SSLCertVerificationError", None, None),
        ),
        excused=False,
    ),
    (DEV, "tls-unexpected-eof"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' requires authentication. Set HF_TOKEN environment variable or run: huggingface-cli login"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("requests.exceptions.SSLError", None, "context"),
            ("urllib3.exceptions.MaxRetryError", None, "cause"),
            ("urllib3.exceptions.SSLError", None, "context"),
            ("ssl.SSLEOFError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "corrupt-tokenizer"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: It looks like the config file at '<cache>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/config.json' is not a valid JSON file."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "context"),
            ("json.decoder.JSONDecodeError", None, "context"),
            ("builtins.StopIteration", None, "context"),
            ("builtins.OSError", None, "context"),
            ("json.decoder.JSONDecodeError", None, "context"),
            ("builtins.StopIteration", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "get-429"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: There was a specific connection error when trying to load Qwen/Qwen2.5-7B-Instruct:\n"
            "429 Too Many Requests for url: http://127.0.0.1:35891/api/resolve-cache/models/Qwen/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28/config.json.\n"
            "stand-in injected 429"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("httpx.HTTPStatusError", 429, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("httpx.HTTPStatusError", 429, None),
        ),
        excused=True,
    ),
    (RELEASE, "get-503"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: There was a specific connection error when trying to load Qwen/Qwen2.5-7B-Instruct:\n"
            "Server error '503 Service Unavailable' for url 'http://127.0.0.1:36693/api/resolve-cache/models/Qwen/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28/config.json'\n"
            "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503\n"
            "\n"
            "stand-in injected 503"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 503, "cause"),
            ("httpx.HTTPStatusError", 503, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 503, "cause"),
            ("httpx.HTTPStatusError", 503, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-401"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' requires authentication. Set HF_TOKEN environment variable or run: huggingface-cli login"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 401, "cause"),
            ("httpx.HTTPStatusError", 401, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 401, "cause"),
            ("httpx.HTTPStatusError", 401, None),
        ),
        excused=False,
    ),
    (RELEASE, "head-403-tagged"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' is gated on Hugging Face. Accept the license and provide HF_TOKEN or run: huggingface-cli login"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.GatedRepoError", 403, "cause"),
            ("httpx.HTTPStatusError", 403, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.GatedRepoError", 403, "cause"),
            ("httpx.HTTPStatusError", 403, None),
        ),
        excused=False,
    ),
    (RELEASE, "head-403-untagged"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:51919' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 403, "cause"),
            ("httpx.HTTPStatusError", 403, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 403, "cause"),
            ("httpx.HTTPStatusError", 403, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-404-tagged"): Capture(
        message=(
            "Model 'Qwen/Qwen2.5-7B-Instruct' not found on HuggingFace. Use the full model path (org/model-name). Try: Qwen/Qwen3-8B, Qwen/Qwen2.5-7B"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.RepositoryNotFoundError", 404, "cause"),
            ("httpx.HTTPStatusError", 404, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.RepositoryNotFoundError", 404, "cause"),
            ("httpx.HTTPStatusError", 404, None),
        ),
        excused=False,
    ),
    (RELEASE, "head-404-untagged"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:51581' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 404, "cause"),
            ("httpx.HTTPStatusError", 404, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 404, "cause"),
            ("httpx.HTTPStatusError", 404, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-429"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:40667' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("httpx.HTTPStatusError", 429, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 429, "cause"),
            ("httpx.HTTPStatusError", 429, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-500"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:32983' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 500, "cause"),
            ("httpx.HTTPStatusError", 500, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 500, "cause"),
            ("httpx.HTTPStatusError", 500, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-503"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:41669' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 503, "cause"),
            ("httpx.HTTPStatusError", 503, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("huggingface_hub.errors.HfHubHTTPError", 503, "cause"),
            ("httpx.HTTPStatusError", 503, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-connection-refused"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:9' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("builtins.ConnectionRefusedError", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("builtins.ConnectionRefusedError", None, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-dns-failure"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://standin-does-not-exist.invalid' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("socket.gaierror", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("socket.gaierror", None, None),
        ),
        excused=True,
    ),
    (RELEASE, "head-timeout"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'http://127.0.0.1:38983' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ReadTimeout", None, "cause"),
            ("httpcore.ReadTimeout", None, "context"),
            ("builtins.TimeoutError", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ReadTimeout", None, "cause"),
            ("httpcore.ReadTimeout", None, "context"),
            ("builtins.TimeoutError", None, None),
        ),
        excused=True,
    ),
    (RELEASE, "offline-cold-cache"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, None),
        ),
        excused=True,
    ),
    (RELEASE, "proxy-answers-cannot-tunnel"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: Can't load the configuration of 'Qwen/Qwen2.5-7B-Instruct'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Qwen/Qwen2.5-7B-Instruct' is the correct path to a directory containing a config.json file"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "context"),
            ("httpx.ProxyError", None, "cause"),
            ("httpcore.ProxyError", None, "context"),
            ("builtins.OSError", None, "context"),
            ("httpx.ProxyError", None, "cause"),
            ("httpcore.ProxyError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "tls-close-notify"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: Can't load the configuration of 'Qwen/Qwen2.5-7B-Instruct'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Qwen/Qwen2.5-7B-Instruct' is the correct path to a directory containing a config.json file"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "context"),
            ("httpx.RemoteProtocolError", None, "cause"),
            ("httpcore.RemoteProtocolError", None, "context"),
            ("builtins.OSError", None, "context"),
            ("httpx.RemoteProtocolError", None, "cause"),
            ("httpcore.RemoteProtocolError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "tls-handshake-alert"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: Can't load the configuration of 'Qwen/Qwen2.5-7B-Instruct'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Qwen/Qwen2.5-7B-Instruct' is the correct path to a directory containing a config.json file"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "context"),
            ("httpx.ReadError", None, "cause"),
            ("httpcore.ReadError", None, "context"),
            ("ssl.SSLError", None, "context"),
            ("builtins.OSError", None, "context"),
            ("httpx.ReadError", None, "cause"),
            ("httpcore.ReadError", None, "context"),
            ("ssl.SSLError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "tls-junk-ca-bundle"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: [X509: NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found (_ssl.c:4416)"
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("ssl.SSLError", None, "context"),
            ("ssl.SSLError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "tls-self-signed-endpoint"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://127.0.0.1:33847' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("ssl.SSLCertVerificationError", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("ssl.SSLCertVerificationError", None, None),
        ),
        excused=False,
    ),
    (RELEASE, "tls-unexpected-eof"): Capture(
        message=(
            "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://127.0.0.1:51361' to load the files, and couldn't find them in the cached files.\n"
            "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ),
        chain=(
            ("builtins.ValueError", None, "cause"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("builtins.ConnectionResetError", None, "context"),
            ("builtins.OSError", None, "cause"),
            ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
            ("httpx.ConnectError", None, "cause"),
            ("httpcore.ConnectError", None, "context"),
            ("ssl.SSLEOFError", None, None),
        ),
        excused=False,
    ),
}


# The one shape the guard cannot tell from a Hub outage, kept here so the gap
# is asserted rather than assumed. See `hub_was_unavailable` for why.
DEAD_PROXY_ON_THE_RELEASE_STACK = Capture(
    message=(
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\n"
        "Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
    ),
    chain=(
        ("builtins.ValueError", None, "cause"),
        ("builtins.OSError", None, "cause"),
        ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
        ("httpx.ConnectError", None, "cause"),
        ("httpcore.ConnectError", None, "context"),
        ("builtins.ConnectionRefusedError", None, "context"),
        ("builtins.OSError", None, "cause"),
        ("huggingface_hub.errors.LocalEntryNotFoundError", None, "cause"),
        ("httpx.ConnectError", None, "cause"),
        ("httpcore.ConnectError", None, "context"),
        ("builtins.ConnectionRefusedError", None, None),
    ),
    excused=True,
)


OUTAGES = {key: capture for key, capture in CAPTURES.items() if capture.excused}
BROKEN = {key: capture for key, capture in CAPTURES.items() if not capture.excused}


def _ids(captures):
    return [f"{environment}: {label}" for environment, label in captures]


@pytest.mark.parametrize("capture", OUTAGES.values(), ids=_ids(OUTAGES))
def test_a_hub_outage_is_excused(capture):
    """Every one of these must print "skipped" rather than block a release.

    Two of them are excused for a reason worth naming rather than a reason worth being
    glad about. An untagged 403 or 404 is excused because huggingface_hub converts it to
    `LocalEntryNotFoundError` before any status is read, and on the dev stack a TLS-level
    close arrives the same way, with no `ssl` link on the chain to deny it. Both are
    measured behaviour rather than intent, and they are here so that if either stops
    being true this notices.
    """
    assert smoke_test.hub_was_unavailable(capture.replay())


@pytest.mark.parametrize("capture", BROKEN.values(), ids=_ids(BROKEN))
def test_a_failure_that_is_not_a_hub_outage_still_fails_the_release(capture):
    """A runner that never reached the Hub, and a tokenizer that is genuinely broken.

    The TLS rows are the deliberate part: an unexpected EOF is denied by
    `BROKEN_RUNNER_TYPES` even where `LocalEntryNotFoundError` sits above it offering to
    excuse it, which is the release gate failing closed on TLS rather than guessing
    whether the fault was the runner's or the remote's.
    """
    assert not smoke_test.hub_was_unavailable(capture.replay())


def test_a_dead_proxy_is_not_distinguishable_on_the_release_stack():
    """The guard's one documented gap, asserted so that it stays a known quantity.

    Under httpx a proxy that refuses the connection and the Hub refusing the connection
    produce the same chain, link for link, so this is excused. If a future huggingface_hub
    or httpx starts distinguishing them, this test fails and the gap can be closed.
    """
    assert smoke_test.hub_was_unavailable(DEAD_PROXY_ON_THE_RELEASE_STACK.replay())
    refused_by_the_hub = CAPTURES[(RELEASE, "head-connection-refused")]
    assert DEAD_PROXY_ON_THE_RELEASE_STACK.chain == refused_by_the_hub.chain


def test_the_guard_must_follow_context_too(monkeypatch):
    """Why the link kind is recorded: reading only `__cause__` publishes a MITM'd runner.

    On the release stack the self-signed chain reaches the `ssl` error only across
    `__context__`. A `_causes` that followed `__cause__` alone would stop at httpcore,
    never see the `ssl` link `BROKEN_RUNNER_TYPES` denies, and be left with the
    `LocalEntryNotFoundError` above it -- which excuses. A corpus whose `replay` joined
    every link with `__cause__` could not tell the two walks apart, and the whole suite
    would stay green while the guard let an intercepted release publish.
    """
    intercepted = CAPTURES[(RELEASE, "tls-self-signed-endpoint")]
    assert not smoke_test.hub_was_unavailable(intercepted.replay())

    def cause_only(error):
        links = []
        seen = set()
        current = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            links.append(current)
            current = current.__cause__
        return links

    monkeypatch.setattr(smoke_test, "_causes", cause_only)
    assert smoke_test.hub_was_unavailable(intercepted.replay())


def test_the_corpus_covers_both_resolved_environments():
    """The bug this corpus exists to prevent was measuring only the locked environment.

    A guard pinned to one stack's wording passed its own tests and still failed at
    release, so neither half of the corpus may quietly disappear.
    """
    for environment in (DEV, RELEASE):
        assert any(env == environment for env, _ in OUTAGES), environment
        assert any(env == environment for env, _ in BROKEN), environment


# The transport each environment resolves. The two stacks share none of it, so the classes
# on a chain say which stack the row was really induced on.
ENVIRONMENT_TRANSPORTS = {
    DEV: ("requests.", "urllib3."),
    RELEASE: ("httpx.", "httpcore."),
}


def test_every_row_names_only_its_own_environments_transport():
    """The check above compares a label; this one checks the chain underneath it.

    A label is just a string, so moving a dev row under the release key would keep
    `test_the_corpus_covers_both_resolved_environments` green while the release half
    quietly stopped being measured. Under requests there is no httpx on the chain and
    under httpx there is no urllib3, so the transport classes are the evidence.
    """
    for environment, other in ((DEV, RELEASE), (RELEASE, DEV)):
        rows = {
            label: capture
            for (env, label), capture in CAPTURES.items()
            if env == environment
        }
        assert rows, environment
        alien = ENVIRONMENT_TRANSPORTS[other]
        for label, capture in rows.items():
            trespassing = [
                name for name, _, _ in capture.chain if name.startswith(alien)
            ]
            assert not trespassing, (environment, label, trespassing)
        own = ENVIRONMENT_TRANSPORTS[environment]
        carried = [
            name
            for capture in rows.values()
            for name, _, _ in capture.chain
            if name.startswith(own)
        ]
        assert carried, environment


def test_the_two_environments_word_the_same_outage_differently():
    """Why the guard reads classes: the prose is not stable across the two stacks.

    Both messages below are the same 429, and no substring of one that identifies it as
    a rate limit appears in the other.
    """
    dev = CAPTURES[(DEV, "get-429")].message
    release = CAPTURES[(RELEASE, "get-429")].message
    assert "429" in dev
    assert "429" in release
    assert dev != release
    assert "client error" in dev.lower()
    assert "client error" not in release.lower()


def _drop_outage_rule(rule):
    def disable(monkeypatch):
        monkeypatch.setattr(
            smoke_test,
            "HUB_OUTAGE_RULES",
            tuple(kept for kept in smoke_test.HUB_OUTAGE_RULES if kept is not rule),
        )

    return disable


def _drop_broken_runner_type(error_type):
    def disable(monkeypatch):
        monkeypatch.setattr(
            smoke_test,
            "BROKEN_RUNNER_TYPES",
            tuple(
                kept
                for kept in smoke_test.BROKEN_RUNNER_TYPES
                if kept is not error_type
            ),
        )

    return disable


# Every independent decision the guard makes, and how to switch it off. Both halves
# belong in one table because the invariant is the same for both: the guard should
# contain nothing that no measured failure needs.
GUARD_RULES = {
    **{rule.__name__: _drop_outage_rule(rule) for rule in smoke_test.HUB_OUTAGE_RULES},
    **{
        f"denies {error_type.__module__}.{error_type.__qualname__}": (
            _drop_broken_runner_type(error_type)
        )
        for error_type in smoke_test.BROKEN_RUNNER_TYPES
    },
}


@pytest.mark.parametrize("disable", GUARD_RULES.values(), ids=GUARD_RULES)
def test_every_rule_earns_its_place(disable, monkeypatch):
    """Turning any one rule off must change the answer for at least one measured failure.

    Each rule is a way a broken release publishes or a healthy one is blocked, so a rule
    that no capture depends on is one nothing measured asked for, and should go.
    """
    disable(monkeypatch)
    assert any(
        smoke_test.hub_was_unavailable(capture.replay()) != capture.excused
        for capture in CAPTURES.values()
    )


def test_a_chain_too_long_to_read_is_not_excused(monkeypatch):
    """Overrunning the bound must fail the release, not excuse it on a partial view."""
    outage = CAPTURES[(RELEASE, "head-429")].replay()
    monkeypatch.setattr(smoke_test, "_CHAIN_LIMIT", 1)
    assert not smoke_test.hub_was_unavailable(outage)


class _UnexpectedResponseError(ValueError):
    """A transport error whose `response` holds something the guard did not expect."""

    response: SimpleNamespace


def test_an_unhashable_status_code_does_not_crash_the_guard():
    """`response.status_code` is whatever the object carries, and the guard must answer.

    Nothing measured produces one, but a set membership test on an unhashable value
    raises TypeError, and a guard that raises out of a release step fails it for a reason
    that has nothing to do with the Hub.
    """
    error = _UnexpectedResponseError("replayed")
    error.response = SimpleNamespace(status_code=[503])
    assert not smoke_test.hub_was_unavailable(error)


def test_a_chain_that_points_back_at_itself_terminates():
    """`__context__` can close a loop, and the guard still has to answer."""
    first = ValueError("first")
    second = ValueError("second")
    first.__context__ = second
    second.__context__ = first
    assert not smoke_test.hub_was_unavailable(first)


def test_the_isolation_restores_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    with smoke_test.isolated_cache_home():
        # setdefault, so a caller that already said where keeps its answer.
        assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)
        created = Path(os.environ["XDG_CONFIG_HOME"])
        assert created != tmp_path

    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)
    assert "XDG_CONFIG_HOME" not in os.environ
    assert not created.exists()


@pytest.mark.slow
def test_the_whole_smoke_test_passes():
    """The release invocation itself, minus the installed artifact."""
    smoke_test.main()
