import numpy as np

# import torch
import torchvision.transforms.functional as F


class SquarePad:
    """ """

    def __init__(self, resize_to: int | None = None):
        self.resize_to = resize_to

    def __call__(self, image):
        w, h = image.size
        if self.resize_to:
            max_wh = self.resize_to
        else:
            max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")

    def __repr__(self):
        return self.__class__.__name__ + "()"
