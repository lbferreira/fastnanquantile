import importlib

from .fastnanquantile import nanquantile

try:
    __version__ = importlib.metadata.version("fastnanquantile") or "unknown"
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
