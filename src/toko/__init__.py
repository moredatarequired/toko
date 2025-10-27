from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("toko")
except PackageNotFoundError:  # e.g. running from a repo checkout without install
    __version__ = "0.0.0"
