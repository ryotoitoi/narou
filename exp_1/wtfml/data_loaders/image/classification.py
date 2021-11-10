#%%
"""
__author__: Abhishek Thakur
"""
from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from transformers import BertJapaneseTokenizer
import transformers
from wtfml.data_loaders.nlp.utils import clean_sentence

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


albumentations_dict = {
    "train": A.Compose(
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
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
            # fixed_image_standardization,
        ]
    ),
    "val": A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    ),
}


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


class ClassificationWithBERTDataset:
    """
    BERTのfeature vectorと合わせてクラシフィケーションタスクを行う場合のDataset #TODO 今日はここ使う
    """

    def __init__(
        self,
        category_ids: np.ndarray,
        #bert_features: np.ndarray,
        input_texts,
        image_paths: str,
        targets: Union[pd.DataFrame, pd.Series, np.ndarray],
        resize: Optional[Tuple] = (224, 224),
        albumentations_dict=albumentations_dict,
        max_len=144,
        backend: str = "pil",
        mode: str = "train",
        clearning_function=clean_sentence,
    ):
        """
        Args:
            category_ids: SQLで取得したカテゴリーidの配列
            bert_features: csvから読み込まれたBERTの特徴量の文字列を含む配列
            image_paths: 画像URLのリスト
            targets: 各ツイートがis_importantかどうかのbool値を含む配列
            resize: リサイズのサイズを指定するtuple、または指定なしのNone
            augmentations: albumentationsのaugmentation
            backend: 画像処理方法を指定するstr。"pil"か"cv2"のみ。
        """
        self.category_ids = category_ids
        self.input_texts = input_texts
        #         self.bert_features = bert_features
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations_dict = albumentations_dict
        self.backend = backend
        self.max_len = max_len
        self.mode = mode
        self.clearning_function = clearning_function
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )

        if mode not in ["train", "val"]:
            raise ValueError('You have to set mode "train" or "val"')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # is_important
        is_important = [1] if self.targets[item] else [0]


        input_text = str(self.input_texts[item])
        if self.clearning_function:
            input_text = self.clearning_function(input_text)

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # return_tensors="pt"
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # category_ids
        category_ids = self.category_ids[item] 

        # 画像の処理
        ## ダミー画像を取得する処理、本番では要修正
        image_path = self.image_paths[item]
        #         image_path = "/home/jupyter/data/xwire-ml-artifacts/kurimonaca/data/train/images/1/2021-09-13/23/332044197/E_NButMUcAAtpQD.jpg"
        if self.backend == "pil":
            image = Image.open(image_path)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if self.resize is not None:
                image = image.resize(
                    (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                )
            image = np.array(image)
        elif self.backend == "cv2":
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[1], self.resize[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
        else:
            raise Exception("Backend not implemented")
        if albumentations_dict is not None:
            augmented = self.augmentations_dict[self.mode](image=image)
            image = augmented["image"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "is_important": torch.tensor(is_important, dtype=torch.long),
            "category": torch.tensor(category_ids, dtype=torch.long),
            "image": image.to(torch.float),
        }


# TODO : distilbertが動くように修正が必要
class ClassificationWithDistilBERTDataset:
    """
    BERTのfeature vectorと合わせてクラシフィケーションタスクを行う場合のDataset #TODO 今日はここ使う
    """

    def __init__(
        self,
        category_ids: np.ndarray,
        #bert_features: np.ndarray,
        input_texts,
        image_paths: str,
        targets: Union[pd.DataFrame, pd.Series, np.ndarray],
        resize: Optional[Tuple] = (224, 224),
        albumentations_dict=albumentations_dict,
        max_len=144,
        backend: str = "pil",
        mode: str = "train",
        clearning_function=clean_sentence,
    ):
        """
        Args:
            category_ids: SQLで取得したカテゴリーidの配列
            bert_features: csvから読み込まれたBERTの特徴量の文字列を含む配列
            image_paths: 画像URLのリスト
            targets: 各ツイートがis_importantかどうかのbool値を含む配列
            resize: リサイズのサイズを指定するtuple、または指定なしのNone
            augmentations: albumentationsのaugmentation
            backend: 画像処理方法を指定するstr。"pil"か"cv2"のみ。
        """
        self.category_ids = category_ids
        self.input_texts = input_texts
        #         self.bert_features = bert_features
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations_dict = albumentations_dict
        self.backend = backend
        self.max_len = max_len
        self.mode = mode
        self.clearning_function = clearning_function
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )

        if mode not in ["train", "val"]:
            raise ValueError('You have to set mode "train" or "val"')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # is_important
        is_important = [1] if self.targets[item] else [0]

        # bert_features: str→tensorへの変換をする #FIXME　ここを直して！
        input_text = str(self.input_texts[item])
        if self.clearning_function:
            input_text = self.clearning_function(input_text)

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # return_tensors="pt"
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        category_ids = self.category_ids[item] 

        # 画像の処理
        ## ダミー画像を取得する処理、本番では要修正
        image_path = self.image_paths[item]
        #         image_path = "/home/jupyter/data/xwire-ml-artifacts/kurimonaca/data/train/images/1/2021-09-13/23/332044197/E_NButMUcAAtpQD.jpg"
        if self.backend == "pil":
            image = Image.open(image_path)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if self.resize is not None:
                image = image.resize(
                    (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                )
            image = np.array(image)
        elif self.backend == "cv2":
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[1], self.resize[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
        else:
            raise Exception("Backend not implemented")
        if albumentations_dict is not None:
            augmented = self.augmentations_dict[self.mode](image=image)
            image = augmented["image"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "is_important": torch.tensor(is_important, dtype=torch.long),
            "category": torch.tensor(category_ids, dtype=torch.long),
            "image": image.to(torch.float),
        }