import os
import shutil
import sys

import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision_utils import create_structure

home = os.environ.get("DETOXAI_DATASET_PATH", os.path.expanduser("~"))
directory = os.path.join(home, "caltech101")

BASE_DIR = directory
TMP_DIR = os.path.join(BASE_DIR, "tmp")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

train_dataset = torchvision.datasets.Caltech101(
    root=TMP_DIR, download=True, transform=transforms.ToTensor()
)
create_structure(train_dataset, BASE_DIR, DATA_DIR)

# Remove the tmp directory
shutil.rmtree(TMP_DIR)
