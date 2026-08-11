from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toko.counter import count_tokens
    from toko.result import TokenCount

try:
    __version__ = version("toko")
except PackageNotFoundError:  # e.g. running from a repo checkout without install
    __version__ = "0.0.0"

__all__ = ["TokenCount", "__version__", "count_tokens"]

# Resolved on first attribute access rather than at import. Reaching count_tokens pulls
# in tiktoken, which triples the cost of an `import toko` that only wanted __version__.
_LAZY_EXPORTS = {"count_tokens": "toko.counter", "TokenCount": "toko.result"}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
