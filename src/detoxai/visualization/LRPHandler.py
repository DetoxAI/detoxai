from functools import partial

import numpy as np
import torch
from torch import nn
from zennit.attribution import (  # noqa
    Attributor,
    Gradient,
    IntegratedGradients,
    Occlusion,
    SmoothGrad,
)
from zennit.canonizers import Canonizer, SequentialMergeBatchNorm  # noqa
from zennit.composites import (  # noqa
    Composite,
    EpsilonAlpha2Beta1,
    EpsilonGammaBox,
    EpsilonPlus,
    EpsilonPlusFlat,
    MixedComposite,
)

from .utils import get_nth_batch

SUPPORTED_CANONIZERS = ["SequentialMergeBatchNorm"]
SUPPORTED_COMPOSITES = [
    "EpsilonPlus",
    "EpsilonAlpha2Beta1",
    "EpsilonPlusFlat",
    "EpsilonGammaBox",
    "MixedComposite",
    None,
]
SUPPORTED_ATTRIBUTTORS = ["Gradient", "SmoothGrad", "IntegratedGradients", "Occlusion"]


class LRPHandler:
    """LRPHandler is a class that handles the calculation of input image attributions for a given model and dataset."""

    def __init__(
        self,
        attributor_name: str = "Gradient",
        composite_name: str = "EpsilonPlus",
        canonizers: list[str] = [],
        n_classes: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LRPHandler

        Parameters:
            - `attributor_name` (str): The name of the attributor to use for LRP
            - `composite_name` (str): The name of the composite to use for LRP
            - `canonizers` (list[str]): The list of canonizers to use
            - `**kwargs`: Additional keyword arguments to pass to the composite or attributor
            
        """
        self.composite_name = composite_name
        self.canonizers = [self.__get_canonizer(c) for c in canonizers]
        self.attributor_name = attributor_name
        self.kwargs = kwargs
        self.n_classes = n_classes

        self.composite = self.__get_composite(**kwargs)

    def calculate(
        self,
        model: nn.Module,
        data_loader: object,
        batch_num: int | None,
        max_images: int | None = None,
    ) -> torch.Tensor:
        """Calculate LRP attribution

        Args:
          model: nn
          data_loader: WrappedDataLoader | torch Dataloader
          batch_num: int
          attribution: is calculated for the entire datset
          model: nn.Module:
          data_loader: object:
          batch_num: int | None:
          max_images: int | None:  (Default value = None)

        Returns:
          - `torch.Tensor` of size (L, N, IMG_w, IMG_h), where N is the number of samples, L is the class on which
          LRP was conditioned, IMG_w and IMG_h are the width and height of the image.

        """
        # Figure out the shape of the tensor to return
        if not hasattr(data_loader, "get_num_classes"):
            raise ValueError(
                """Data loader must have a method get_num_classes() to get the number of classes. Preferably, use a `WrappedDataLoader` from detoxai.utils.dataloader.
                Alternatively, you can pass the number of classes as an argument to the LRPHandler constructor."""
            )
        else:
            L = data_loader.get_num_classes()
        
        batch_shape = next(iter(data_loader))[0].shape

        model_device = next(model.parameters()).device

        N = 1e9

        if max_images is not None:
            N = min(max_images, N)  # Max number of images to process
        N = min(N, batch_shape[0])  # Can't process more images than in the batch

        shape = (L, N, batch_shape[3], batch_shape[2])
        imgs_w_attribution = torch.zeros(shape, device=model_device)

        def attr_output_fn(output, target, num_classes):
            """

            Args:
              output:
              target:
              num_classes:

            Returns:

            """
            # output times one-hot encoding of the target labels of size (len(target), 1000)
            return output * nn.functional.one_hot(target, num_classes=num_classes)

        with self.__get_attributor(model) as attributor:
            if batch_num is None:
                raise NotImplementedError()
            else:
                # Get a proper batch and calculate LRP
                batched_img, _, _ = get_nth_batch(data_loader, batch_num)

                # Take only N images
                batched_img = batched_img[:N]
                batched_img = batched_img.to(model_device)

                for _label in range(L):
                    labels = torch.tensor([_label], device=model_device)
                    output_relevance = partial(
                        attr_output_fn, target=labels, num_classes=L
                    )

                    out, relevance = attributor(batched_img, output_relevance)

                    relevance = relevance.sum(1).detach().cpu().numpy()
                    amax = np.abs(relevance).max((1, 2), keepdims=True)
                    relevance = (relevance + amax) / 2 / amax

                    imgs_w_attribution[_label] = torch.Tensor(relevance)
        return imgs_w_attribution

    def __get_composite(self, **kwargs) -> Composite:
        """
        Resolve and instantiate the composite class

        Returns:
            - `Composite` instance
        """
        if self.composite_name is None:
            return None

        if self.composite_name in SUPPORTED_COMPOSITES:
            composite_class = globals().get(self.composite_name)
            if composite_class is None:
                raise ValueError(f"Composite class {self.composite_name} not found")

        composite = composite_class(canonizers=self.canonizers, **kwargs)

        return composite

    def __get_attributor(self, model: nn.Module) -> Attributor:
        """
        Resolve and instantiate the attributor class

        Parameters:
            - `model` (nn.Module): The model to use for LRP

        Returns:
            - `Attributor` instance
        """
        if self.attributor_name in SUPPORTED_ATTRIBUTTORS:
            attributor_class = globals().get(self.attributor_name)
            if attributor_class is None:
                raise ValueError(f"Attributor class {self.attributor_name} not found")

        if self.composite:
            attributor = attributor_class(model, composite=self.composite)
        else:
            attributor = attributor_class(model)

        return attributor

    def __get_canonizer(self, canonizer_name: str) -> Canonizer:
        """
        Resolve and instantiate the canonizer class

        Parameters:
            - `canonizer_name` (str): The name of the canonizer to use

        Returns:
            - `Canonizer` instance
        """
        if canonizer_name in SUPPORTED_CANONIZERS:
            canonizer_class = globals().get(canonizer_name)
            if canonizer_class is None:
                raise ValueError(f"Canonizer class {canonizer_name} not found")

        canonizer = canonizer_class()

        return canonizer
