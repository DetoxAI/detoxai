import os
import shutil
import zipfile

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

home = os.environ.get("DETOXAI_DATASET_PATH", os.path.expanduser("~"))
directory = os.path.join(home, "fairface")

tmp_directory = os.path.join(directory, "tmp")
data_path = os.path.join(tmp_directory, "data.zip")
uzip_directory = os.path.join(tmp_directory, "data")


# Extract data to directory
with zipfile.ZipFile(data_path, "r") as zip_ref:
    zip_ref.extractall(uzip_directory)
    print("Done")

# Read csvs
l_train_df = pd.read_csv(os.path.join(tmp_directory, "l_train.csv"))
l_val_df = pd.read_csv(os.path.join(tmp_directory, "l_val.csv"))

concatenated = pd.concat([l_train_df, l_val_df]).reset_index(drop=True)


columns_to_encode = ["age", "gender", "race", "service_test"]
mapping = {}

for column in columns_to_encode:
    concatenated[column] = concatenated[column].astype(str)
    label_encoder = LabelEncoder()

    concatenated[column] = label_encoder.fit_transform(concatenated[column].astype(str))

    mapping[column] = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

reverse_mapping = {
    str(column): {int(value): str(key) for key, value in mapping[column].items()}
    for column in mapping
}

mapping_path = os.path.join(directory, "labels_mapping.yaml")
with open(mapping_path, "w") as f:
    yaml.dump(reverse_mapping, f)

# Change in column 'file'
concatenated["file"] = concatenated["file"].apply(lambda x: x.split("/")[1])

# Save concatenated csv
concatenated.to_csv(os.path.join(directory, "labels.csv"), index=False)


# Move data from tmp to directory under data/
# it has now valid and train subdirectories but we don't need them
# so we will move the files to the data directory

data_directory = os.path.join(directory, "data")
os.makedirs(data_directory, exist_ok=True)

# Move files
counter = 0
for subdir, dirs, files in os.walk(uzip_directory):
    for file in files:
        source, extension = os.path.splitext(file)
        target = os.path.join(data_directory, f"{counter}{extension}")
        shutil.move(os.path.join(subdir, file), os.path.join(data_directory, target))
        counter += 1

# Remove tmp directory
shutil.rmtree(tmp_directory)
