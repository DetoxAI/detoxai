from enum import Enum

import torch.nn as nn
import torchvision.models as models


class SupportedModels(Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    EFFICIENTNET_B6 = "efficientnet_b6"
    EFFICIENTNET_B7 = "efficientnet_b7"
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"
    CONVNEXT_LARGE = "convnext_large"
    INCEPTION_V3 = "inception_v3"
    WIDE_RESNET50_2 = "wide_resnet50_2"
    WIDE_RESNET101_2 = "wide_resnet101_2"
    EFFICIENTNET_V2_S = "efficientnet_v2_s"
    EFFICIENTNET_V2_M = "efficientnet_v2_m"
    EFFICIENTNET_V2_L = "efficientnet_v2_l"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"
    MAXVIT_T = "maxvit_t"
    VIT_B_32 = "vit_b_32"


class PretrainedModel(nn.Module):
    def __init__(
        self, num_classes: int, model_name: SupportedModels | str, device: str = None
    ) -> None:
        super(PretrainedModel, self).__init__()
        self.num_classes = num_classes

        if isinstance(model_name, str):
            model_name = SupportedModels(model_name)

        self.model_name = model_name

        self._get_pretrained_model()
        self._change_output_layer()

        if device:
            self.to(device)

    def forward(self, x):
        return self.model(x)

    def _get_pretrained_model(self):
        """
        Autoresolve the model based on the model_name by using pattern matching
        and model family grouping.
        """
        model_name = self.model_name.value
        self.model = getattr(models, model_name)(pretrained=True)

    def _change_output_layer(self):
        """
        Change the output layer of the model to match the number of classes
        based on model architecture.
        """
        model_name = self.model_name.value

        print(self.model)

        if model_name.startswith("resnet"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("efficientnet"):
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("vgg"):
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("mobilenet_v3"):
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("convnext"):
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("inception"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("wide_resnet"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("efficientnet_v2"):
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("maxvit"):
            in_features = self.model.classifier[5].in_features
            self.model.classifier[5] = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("swin_v2"):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, self.num_classes)

        elif model_name.startswith("vit"):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, self.num_classes)
