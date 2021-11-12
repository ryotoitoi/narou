from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


albumentations = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.HueSaturationValue(),
        A.RGBShift(),
        A.RandomGamma(),
        # A.GaussianBlur(p=0.3, blur_limit=(3, 3)),
        # A.ElasticTransform(),
        # A.ShiftScaleRotate(),
        A.ToGray(p=0.2),
        A.ImageCompression(quality_lower=95, p=0.3),
        # A.CLAHE(tile_grid_size=(4, 4)),
        # A.ISONoise(),
        # A.HorizontalFlip(p=0.5),
        # A.Resize(height=160, width=160, interpolation=cv2.INTER_LANCZOS4, p=1),
        # ToTensorV2(),
        # fixed_image_standardization,
    ]
)


class ClassificationDataset:
    """
    クラシフィケーションタスク用のDataset
    """

    def __init__(
        self,
        image_paths: str,
        targets: Union[pd.DataFrame, pd.Series, np.ndarray],
        resize: Optional[Tuple],
        augmentations=None,
        backend: str = "pil",
    ):
        """
        Args:
            :param image_paths: list of paths to images
            :param targets: numpy array
            :param resize: tuple or None
            :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            if self.resize is not None:
                image = image.resize(
                    (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                )
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        elif self.backend == "cv2":
            image = cv2.imread(self.image_paths[item])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[1], self.resize[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
            image = augmented["image"]
        else:
            raise Exception("Backend not implemented")

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }


class ClassificationDataLoader:
    def __init__(
        self,
        image_paths: str,
        targets: Union[pd.DataFrame, pd.Series, np.ndarray],
        resize: Optional[Tuple],
        augmentations=None,
        backend: str = "pil",
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.dataset = ClassificationDataset(
            image_paths=self.image_paths,
            targets=self.targets,
            resize=self.resize,
            augmentations=self.augmentations,
            backend=self.backend,
            channel_first=self.channel_first,
        )

    def fetch(
        self,
        batch_size: int,
        num_workers: int,
        drop_last: bool = False,
        shuffle: bool = True,
        tpu: bool = False,
        sampler=None,
    ):
        """
        :param batch_size: batch size
        :param num_workers: number of processes to use
        :param drop_last: drop the last batch?
        :param shuffle: True/False
        :param tpu: True/False, to use tpu or not
        """
        if tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return data_loader
