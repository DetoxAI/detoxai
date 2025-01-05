import os
import shutil
import zipfile

import pandas as pd
import yaml

home = os.environ.get("DETOXAI_DATASET_PATH", os.path.expanduser("~"))
directory = os.path.join(home, "celeba")

tmp_directory = os.path.join(directory, "tmp")
data_path = os.path.join(tmp_directory, "kaggle.zip")


# Extract data to directory
with zipfile.ZipFile(data_path, "r") as zip_ref:
    zip_ref.extractall(tmp_directory)
    print("Done")

# Read csvs
df = pd.read_csv(os.path.join(tmp_directory, "list_attr_celeba.csv"))

# Transform all -1 to 0
df = df.replace(-1, 0)

# Create mapping for each attribute 1 - present, 0 - not present
mapping = {column: {1: "present", 0: "not present"} for column in df.columns[1:]}


mapping_path = os.path.join(directory, "labels_mapping.yaml")
with open(mapping_path, "w") as f:
    yaml.dump(mapping, f)

# Save concatenated csv
df.to_csv(os.path.join(directory, "labels.csv"), index=False)


# Move data from tmp to directory under data/
# it has now valid and train subdirectories but we don't need them
# so we will move the files to the data directory

data_directory = os.path.join(directory, "data")
os.makedirs(data_directory, exist_ok=True)

# Move files
img_dir = os.path.join(tmp_directory, "img_align_celeba")
for subdir, dirs, files in os.walk(img_dir):
    for file in files:
        source, extension = os.path.splitext(file)
        target = os.path.join(data_directory, f"{source.zfill(6)}{extension}")
        shutil.move(os.path.join(subdir, file), os.path.join(data_directory, target))

# Remove tmp directory
shutil.rmtree(tmp_directory)
