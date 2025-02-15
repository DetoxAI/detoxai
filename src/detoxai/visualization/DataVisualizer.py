import numpy as np
import torch

from ..utils.dataloader import DetoxaiDataLoader
from .ImageVisualizer import ImageVisualizer


class DataVisualizer(ImageVisualizer):
    def __init__(self, data_loader: DetoxaiDataLoader, plot_config: dict = {}) -> None:
        self.data_loader = data_loader
        self.set_up_plots_configuration(plot_config)

    def visualize_batch(
        self,
        batch_num: int,
        max_images: int | None = 36,
        return_fig: bool = False,
        batch_preds: torch.Tensor | None = None,
        show_labels: bool = True,
    ) -> None:
        images, labels, prot_attr = self.data_loader.get_nth_batch(batch_num)

        # Check if the images are in the correct format (numpy)
        if isinstance(images, torch.Tensor):
            images = images.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        if max_images is None:
            max_images = images.shape[0]

        images_to_show = min(images.shape[0], max_images)
        rows = int(images_to_show**0.5)
        cols = int(images_to_show**0.5)

        fig, ax = self.get_canvas(
            rows=rows,
            cols=cols,
            shape=(
                int(rows) * self.plot_shape_multiplier,
                int(cols) * self.plot_shape_multiplier,
            ),
        )

        for i, img in enumerate(images[:max_images]):
            assert isinstance(img, np.ndarray), "Image must be a numpy array"

            if img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.shape[0] == 3:
                # img = img.permute(1, 2, 0)
                img = img.transpose((1, 2, 0))
            ax[i // cols, i % cols].imshow(img, vmin=0, vmax=1)

            if show_labels:
                # Add label in the upper left corner
                ax[i // cols, i % cols].text(
                    0.05,
                    0.05,
                    f"Label: {labels[i].item()}",
                    color="green",
                    transform=ax[i // cols, i % cols].transAxes,
                    fontsize=self.fontsize,
                    fontweight="bold",
                )

                # Add prediction in the upper right corner
                if batch_preds is not None:
                    ax[i // cols, i % cols].text(
                        0.05,
                        0.15,
                        f"Pred: {batch_preds[i].item()}",
                        color="red",
                        transform=ax[i // cols, i % cols].transAxes,
                        fontsize=self.fontsize,
                        fontweight="bold",
                    )

    def visualize_agg(self, batch_num: int) -> None:
        images, labels, prot_attr = self.data_loader.get_nth_batch(batch_num)

        assert isinstance(images, torch.Tensor), "Images must be a tensor"
        assert isinstance(labels, torch.Tensor), "Labels must be a tensor"
        assert isinstance(prot_attr, torch.Tensor), (
            "Protected attributes must be a tensor"
        )

        ulab = labels.unique()
        uprot = prot_attr.unique()

        fig, ax = self.get_canvas(
            rows=len(ulab), cols=len(uprot), shape=(len(ulab) * 3, len(uprot) * 3)
        )

        for row, label in enumerate(ulab):
            for col, prot_a in enumerate(uprot):
                mask = (labels == label) & (prot_attr == prot_a)

                img = images[mask].mean(dim=0).cpu().detach().numpy()
                ax[row, col].imshow(img.transpose((1, 2, 0)))
