"""Test support for reaching the Hugging Face Hub, which rate-limits anonymous callers.

`transformers.AutoTokenizer.from_pretrained` asks the Hub for model metadata on every
load — `_patch_mistral_regex` calls `model_info()`, whose response is never cached — so
even a fully warm `HF_HOME` still needs the Hub to answer. Anonymous requests share a
per-IP quota, so a run can be refused with HTTP 429 while the code under test is fine.
"""

import contextlib

import pytest
import requests.adapters

HUB_HOST = "huggingface.co"


@contextlib.contextmanager
def skip_if_rate_limited():
    """Skip the test if the Hub answers 429 while the block runs.

    Detection is on the HTTP status rather than on an error message, so an unreachable
    Hub, a missing model and a broken tokenizer all still fail the test.
    """
    original_send = requests.adapters.HTTPAdapter.send
    refused: list[str] = []

    def send(self, request, *args, **kwargs):
        response = original_send(self, request, *args, **kwargs)
        if response.status_code == 429 and HUB_HOST in str(request.url):
            refused.append(str(request.url))
        return response

    patch = pytest.MonkeyPatch()
    patch.setattr(requests.adapters.HTTPAdapter, "send", send)
    try:
        yield
    finally:
        patch.undo()
        if refused:
            pytest.skip(f"Hugging Face Hub rate limit (429) on {refused[0]}")
