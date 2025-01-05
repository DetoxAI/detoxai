import os
from pathlib import Path

DETOXAI_ROOT_PATH = Path(os.path.expanduser("~")) / ".detoxai"
DETOXAI_DATASET_PATH = DETOXAI_ROOT_PATH / "datasets"

os.environ["DETOXAI_ROOT_PATH"] = str(DETOXAI_ROOT_PATH)
os.environ["DETOXAI_DATASET_PATH"] = str(DETOXAI_DATASET_PATH)

from .datasets.catalog.download import download_datasets  # noqa
from .core.interface import debias  # noqa

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
