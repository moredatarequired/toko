from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toko.counter import count_tokens
    from toko.result import Caveat, CaveatKind, Retirement, TokenCount

try:
    __version__ = version("toko")
except PackageNotFoundError:  # e.g. running from a repo checkout without install
    __version__ = "0.0.0"

__all__ = [
    "Caveat",
    "CaveatKind",
    "Retirement",
    "TokenCount",
    "__version__",
    "count_tokens",
]


# The lazy names are not in the namespace until first access, so without this
# dir(toko) omits them and lists the import machinery this module happens to use.
# Unioning with globals() keeps the module dunders that dir-based tooling expects.
def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


# Resolved on first attribute access rather than at import. Reaching count_tokens pulls
# in tiktoken, which triples the cost of an `import toko` that only wanted __version__.
_LAZY_EXPORTS = {
    "count_tokens": "toko.counter",
    "Caveat": "toko.result",
    "CaveatKind": "toko.result",
    "Retirement": "toko.result",
    "TokenCount": "toko.result",
}


# Hidden from type checkers: a visible module __getattr__ is a catch-all that types
# every unknown toko.<name> as object, so `toko.TokenCnt` would pass a downstream
# check and only fail at runtime. The TYPE_CHECKING imports above are the real names.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        module_name = _LAZY_EXPORTS.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return getattr(import_module(module_name), name)
