import importlib.util as iu
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Union

import gdown
import yaml

DETOXAI_DATASET_PATH = os.environ.get("DETOXAI_DATASET_PATH", os.path.expanduser("~"))

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = ["cifar10", "cifar100", "caltech101", "celeba", "fairface"]


# Define download function
def gdown_download(url, output, quiet=False):
    """

    Args:
      url:
      output:
      quiet:  (Default value = False)

    Returns:

    """
    logger.debug(f"Downloading {url} to {output}...")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=quiet, fuzzy=False)
    return output


def curl_download(url, output, quiet=False):
    """

    Args:
      url:
      output:
      quiet:  (Default value = False)

    Returns:

    """
    logger.debug(f"Downloading {url} to {output}...")
    if not os.path.exists(output):
        os.system(f"curl -L {url} -o {output}")
    return output


def download_stuff(folder, links, tmp_dir):
    """

    Args:
      folder:
      links:
      tmp_dir:

    Returns:

    """
    logger.debug(f"Downloading data files for {folder}...")

    all_links = list()

    for link in links.values():
        url = link["url"]
        output = os.path.join(tmp_dir, link["output"])
        type = link["type"]
        all_links.append((url, output, type))

    # If type == google_drive then use gdown_download
    # Else use wget_download, but we need to implement it

    processes = []

    for link in all_links:
        if link[2] == "google_drive":
            p = mp.Process(target=gdown_download, args=(link[0], link[1]))
            p.start()
            processes.append(p)
        elif link[2] == "curl":
            p = mp.Process(target=curl_download, args=(link[0], link[1]))
            p.start()
            processes.append(p)
        else:
            logger.debug(
                f"WARNING: {link} not downloaded as type {link[2]} is not supported yet."
            )

    for p in processes:
        p.join()


def run_handler(folder, dir_path, tmp_dir):
    """

    Args:
      folder:
      dir_path:
      tmp_dir:

    Returns:

    """
    # Now import handler.py from 'folder'
    handler_path = os.path.join(dir_path, "handler.py")
    spec = iu.spec_from_file_location("handler", handler_path)
    handler = iu.module_from_spec(spec)
    spec.loader.exec_module(handler)


def download_datasets(
    datasets: List[str], dataset_path: Union[str, Path] = DETOXAI_DATASET_PATH
) -> None:
    """Downloads datasets from the list and save them in directory specified by dataset_path.

    Args:
      - datasets: List of datasets to download e.g., ['celeba', 'fairface']
      - dataset_path: Path to save the datasets.
      datasets: List[str]:
      dataset_path: Union[str:
      Path]:  (Default value = DETOXAI_DATASET_PATH)

    Returns:

    """
    os.environ["DETOXAI_DATASET_PATH"] = str(dataset_path)

    # Discover local folders in __file__ directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [
        f
        for f in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, f)) and f != "__pycache__"
    ]
    logger.debug(f"Discovered folders: {folders}")
    folders = datasets

    try:
        os.makedirs(dataset_path, exist_ok=True)
    except Exception as e:
        logger.debug(f"Error: {e}")

    for folder in folders:
        code_dirpath = os.path.join(current_dir, folder)
        target_dirpath = os.path.join(dataset_path, folder)
        tmp_dir = os.path.join(target_dirpath, "tmp")

        # if labels.csv and label_mapping.yaml and data folder already exists then skip
        if (
            os.path.exists(os.path.join(target_dirpath, "labels.csv"))
            and os.path.exists(os.path.join(target_dirpath, "labels_mapping.yaml"))
            and os.path.exists(os.path.join(target_dirpath, "data"))
        ):
            logger.info(f"{folder} already exists. Skipping...")
            continue

        os.makedirs(tmp_dir, exist_ok=True)
        links = yaml.safe_load(open(os.path.join(code_dirpath, "links.yaml")))
        logger.debug(f"Downloading files for {folder}...")
        logger.debug(links)

        # If torchvision then no need to pre-download
        if isinstance(links, str) and links == "torchvision":
            run_handler(folder, code_dirpath, tmp_dir)
        else:
            download_stuff(folder, links, tmp_dir)
            run_handler(folder, code_dirpath, tmp_dir)


if __name__ == "__main__":
    download_datasets(SUPPORTED_DATASETS)
