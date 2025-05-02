from detoxai.utils.dataloader import DetoxaiDataLoader
from detoxai.utils.datasets import get_detoxai_datasets, DetoxaiDataset
from detoxai.utils.transformations import SquarePad
from detoxai import download_datasets
import torchvision
import torch
import lightning as L
import torchvision
from detoxai.core.model_wrappers import FairnessLightningWrapper
import gdown

device = "cpu"
L.seed_everything(123, workers=True)


def get_your_dataloaders() -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    download_datasets(["celeba"])

    CELEBA_DATASET_CONFIG = {
        "name": "celeba",
        "target": "Smiling",  # target attribute that should be predicted
        # Note that celeba has quite a lot of examples, so to make things easier,
        # we will use a smaller subset of the dataset
        "splits": {"train": 0.003, "test": 0.01, "unlearn": 0.003, "no": 0.984},
    }

    dataset: list[DetoxaiDataset] = get_detoxai_datasets(
        CELEBA_DATASET_CONFIG,
        transform=torchvision.transforms.Compose(
            [SquarePad(), torchvision.transforms.ToTensor()]
        ),
        device="cpu",
    )

    pa = "Wearing_Hat"  # protected attribute
    pa_value = 1  # protected attribute value

    collate_fn = dataset["train"].get_collate_fn(pa, pa_value)

    dataloader_train = DetoxaiDataLoader(
        dataset["train"], collate_fn=collate_fn, batch_size=256
    )
    dataloader_unlearn = DetoxaiDataLoader(
        dataset["unlearn"], collate_fn=collate_fn, batch_size=128
    )
    dataloader_test = DetoxaiDataLoader(
        dataset["test"], collate_fn=collate_fn, batch_size=256
    )
    return dataloader_test, dataloader_unlearn


def get_your_model() -> torch.nn.Module:
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Make it binary classification

    link = "https://drive.google.com/uc?id=1g9g7Hz5iZbiYvHQB4283QFQy2kMGG6Hf"

    gdown.download(link, "./model.pt")

    od = torch.load("./model.pt")
    od = {k[6:]: v for k, v in od.items()}

    model.load_state_dict(od)

    model = model.to(device)

    return model
