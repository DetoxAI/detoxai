import importlib.util as iu
import multiprocessing as mp
import os

import gdown
import yaml

# Discover local folders in __file__ directory
current_dir = os.path.dirname(os.path.abspath(__file__))
folders = [
    f
    for f in os.listdir(current_dir)
    if os.path.isdir(os.path.join(current_dir, f)) and f != "__pycache__"
]
print(f"Discovered folders: {folders}")


# Define download function
def gdown_download(url, output, quiet=False):
    print(f"Downloading {url} to {output}...")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=quiet, fuzzy=False)
    return output


def curl_download(url, output, quiet=False):
    print(f"Downloading {url} to {output}...")
    if not os.path.exists(output):
        os.system(f"curl -L {url} -o {output}")
    return output


def download_stuff(folder, links, tmp_dir):
    print(f"Downloading data files for {folder}...")

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
            print(
                f"WARNING: {link} not downloaded as type {link[2]} is not supported yet."
            )

    for p in processes:
        p.join()


def run_handler(folder, dir_path, tmp_dir):
    # Now import handler.py from 'folder'
    handler_path = os.path.join(dir_path, "handler.py")
    spec = iu.spec_from_file_location("handler", handler_path)
    handler = iu.module_from_spec(spec)
    spec.loader.exec_module(handler)


for folder in folders:
    dir_path = os.path.join(current_dir, folder)
    tmp_dir = os.path.join(dir_path, "tmp")

    # if labels.csv and label_mapping.yaml and data folder already exists then skip
    if (
        os.path.exists(os.path.join(dir_path, "labels.csv"))
        and os.path.exists(os.path.join(dir_path, "labels_mapping.yaml"))
        and os.path.exists(os.path.join(dir_path, "data"))
    ):
        print(f"{folder} already exists. Skipping...")
        continue

    os.makedirs(tmp_dir, exist_ok=True)
    links = yaml.safe_load(open(os.path.join(dir_path, "links.yaml")))
    print(f"Downloading files for {folder}...")
    print(links)

    # If torchvision then no need to pre-download
    if isinstance(links, str) and links == "torchvision":
        run_handler(folder, dir_path, tmp_dir)
    else:
        download_stuff(folder, links, tmp_dir)
        run_handler(folder, dir_path, tmp_dir)
