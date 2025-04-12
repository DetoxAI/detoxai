import os

import pandas as pd
import torch
import torchvision
import yaml


def create_structure2(train_dataset, test_dataset, base_dir, data_dir) -> None:
    """

    Args:
      train_dataset:
      test_dataset:
      base_dir:
      data_dir:

    Returns:

    """
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    dataloaders = [train_dataloader, test_dataloader]

    __create(dataloaders, train_dataset, base_dir, data_dir)


def create_structure(train_dataset, base_dir, data_dir) -> None:  # noqa
    """

    Args:
      train_dataset:
      base_dir:
      data_dir:

    Returns:

    """
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )
    dataloaders = [train_dataloader]

    __create(dataloaders, train_dataset, base_dir, data_dir)


def __create(dataloaders, train_dataset, base_dir, data_dir) -> None:
    filename_label_map = {}

    # IF hasattr(train_dataset, "classes") then use it else use categories
    if hasattr(train_dataset, "classes"):
        label_name_map = {i: name for i, name in enumerate(train_dataset.classes)}
    elif hasattr(train_dataset, "categories"):
        label_name_map = {i: name for i, name in enumerate(train_dataset.categories)}
    else:
        raise ValueError(
            "Don't panic, our wrapper doesn't support this dataset as it has uncommon attribute mapping in the dataset class. \
            You can easily add support for this dataset by adding a new elif block here in the __create function."
        )

    print(f"Copying files to {data_dir}...")
    # Save all images into /datasets/cifar10/data
    for dataloader in dataloaders:
        for i, data in enumerate(dataloader):
            img, lab = data
            filename = f"{i}.png"
            full_path = os.path.join(data_dir, filename)
            torchvision.utils.save_image(img, full_path, format="png")
            filename_label_map[filename] = label_name_map[lab.item()]

    # Create labels.csv that will ahve filename and one hot encoded labels

    df = pd.DataFrame.from_dict(filename_label_map, orient="index", columns=["label"])
    df = df.reset_index().rename(columns={"index": "filename"})
    df = pd.get_dummies(df, columns=["label"], prefix="", prefix_sep="")
    df = df.astype({col: "int" for col in df.columns if col != "filename"})
    df.to_csv(f"{base_dir}/labels.csv", index=False)

    # Create label_mapping.yaml

    d = {}

    for label in df.columns[1:]:
        d[label] = {0: "not present", 1: "present"}

    with open(f"{base_dir}/labels_mapping.yaml", "w") as f:
        yaml.dump(d, f)
