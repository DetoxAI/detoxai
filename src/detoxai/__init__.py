import importlib.metadata
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.info("Loading DETOXAI...")


# Only if the environment variables are not set
if "DETOXAI_ROOT_PATH" not in os.environ:
    DETOXAI_ROOT_PATH = Path(os.path.expanduser("~")) / ".detoxai"
    logging.info(f"DETOXAI_ROOT_PATH: {DETOXAI_ROOT_PATH}")
    os.environ["DETOXAI_ROOT_PATH"] = str(DETOXAI_ROOT_PATH)

if "DETOXAI_DATASET_PATH" not in os.environ:
    DETOXAI_ROOT_PATH = Path(os.environ["DETOXAI_ROOT_PATH"])
    DETOXAI_DATASET_PATH = DETOXAI_ROOT_PATH / "datasets"
    os.environ["DETOXAI_DATASET_PATH"] = str(DETOXAI_DATASET_PATH)

logger.info(
    f"Detoxai paths: {os.getenv('DETOXAI_ROOT_PATH')}, {os.getenv('DETOXAI_DATASET_PATH')}"
)

# Import all modules in detoxai, must be done after setting the environment variables
from .core.interface import debias, get_supported_methods  # noqa
from .core.results_class import CorrectionResult  # noqa
from .datasets.catalog.download import download_datasets  # noqa

from . import cavs, core, datasets, methods, metrics, utils, visualization  # noqa

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
