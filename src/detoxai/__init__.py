from .datasets.catalog.download import download_datasets  # noqa
from .core.interface import debias  # noqa


import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
