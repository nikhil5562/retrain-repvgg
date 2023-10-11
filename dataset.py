from torch.utils.data import Dataset
from torchvision.transforms._presets import ImageClassification
import albumentations as A
import pandas as pd
import numpy as np
import torch
import cv2
import os

from typing import Callable, Optional

idx_to_label = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
label_to_idx = {label: idx for idx, label in enumerate(idx_to_label)}


class FaceDataset(Dataset):
    def __init__(self, path: str, split: str, augment: bool, transforms: Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
        self.root = path
        self.df = pd.read_csv(os.path.join(self.root, f"{split}.csv"))
        self.transforms = ImageClassification(
            crop_size=256,
            resize_size=224,
            antialias=True
        ) if transforms is None else transforms
        self._augment = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    always_apply=True, contrast_limit=0.2, brightness_limit=0.2
                ),
                A.Cutout(max_h_size=64, max_w_size=64, num_holes=8, p=0.5),
                A.OneOf(
                    [
                        A.MotionBlur(always_apply=True),
                        A.GaussNoise(always_apply=True),
                        A.GaussianBlur(always_apply=True),
                    ],
                    p=0.5,
                ),
                A.Perspective(scale=.1, p=1.),
                A.ColorJitter(p=1., ),
                A.RandomBrightnessContrast(p=1., ),
                A.Downscale(p=.8, scale_min=0.1, scale_max=0.9),
            ]
        ) if augment else None

    def __getitem__(self, idx: int) -> np.ndarray:
        label, path = self.df.loc[idx, ["label", "pth"]]
        img = cv2.imread(os.path.join(self.root, path), cv2.COLOR_BGR2RGB)
        assert img is not None, f"Image at {os.path.join(self.root, path)} is None"
        if self._augment:
            img = self._augment(image=img)["image"]
        img = torch.tensor(img).permute((2, 0, 1))
        if self.transforms:
            img = self.transforms(img)
        return img, label_to_idx[label]

    def __len__(self) -> int:
        return len(self.df)