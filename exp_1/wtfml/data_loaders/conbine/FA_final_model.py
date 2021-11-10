#%%
from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from transformers import BertJapaneseTokenizer
from wtfml.data_loaders.nlp.utils import clean_sentence
from wtfml.data_loaders.utils import weather_dict

#%%
try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_transformers = A.Compose(
    [
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)


def weather2tensor(weather_text: str):
    results = np.zeros(10)
    for key, value in weather_dict.items():
        if key in weather_text:
            results[value] = 1
    if sum(results) == 0:
        results[0] = 1
    return results


class FastAlartFinalClassificationDataset:
    """
    BERTのfeature vectorと合わせてクラシフィケーションタスクを行う場合のDataset
    """

    def __init__(
        self,
        category_ids: np.ndarray,
        input_texts,
        wether_list: Union[List, np.ndarray, pd.Series],  # FIXME ここを修正したい
        is_holyday_list: Union[List, np.ndarray, pd.Series],
        hour_list: Union[List, np.ndarray, pd.Series],
        month_list: Union[List, np.ndarray, pd.Series],
        highest_temperture_list: Union[List, np.ndarray, pd.Series],
        lowest_temperture_list: Union[List, np.ndarray, pd.Series],
        image_paths: str,
        targets: Union[pd.DataFrame, pd.Series, np.ndarray],
        resize: Optional[Tuple] = (128, 128),
        image_transformers=image_transformers,
        max_len=144,
        backend: str = "pil",
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
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.image_transformers = image_transformers
        self.backend = backend
        self.max_len = max_len
        self.clearning_function = clearning_function
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.wether_list = (wether_list,)  # FIXME ここを修正したい
        self.is_holyday_list = is_holyday_list
        self.hour_list = hour_list
        self.month_list = month_list
        self.highest_temperture_list = highest_temperture_list
        self.lowest_temperture_list = lowest_temperture_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
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
        token_type_ids = inputs["token_type_ids"]

        # 画像の処理
        ## ダミー画像を取得する処理、本番では要修正
        image_path = self.image_paths[item]
        if image_path:
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
            if self.image_transformers is not None:
                augmented = self.image_transformers(image=image)
                image = augmented["image"]
        else:
            image = None

        is_important = [1] if self.targets[item] else [0]
        category_ids = self.category_ids[item]

        hours = self.hours_list[item]
        month = self.month_list[item]
        highest_temperture = self.highest_temperture_list[item]
        lowest_temperture = self.lowest_temperture_list[item]
        is_holyday = 1 if self.is_holyday_list[item] else 0
        normalized_hours = hours / 24
        normalized_month = month / 12
        normalized_highest_temperture = highest_temperture / 50
        normalized_lowest_temperture = lowest_temperture / 50
        if self.weather_list[item]:
            weather = weather2tensor(self.weather_list[item])
        else:
            weather = np.zeros(10)
            weather[0] = 1
        metadata_list = [
            normalized_hours,
            normalized_month,
            normalized_highest_temperture,
            normalized_lowest_temperture,
            is_holyday,
        ] + list(weather)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "is_important": torch.tensor(is_important, dtype=torch.long),
            "category": torch.tensor(category_ids, dtype=torch.long),
            "image": image.to(torch.float) if image else None,
            "metadata": torch.tensor(metadata_list, dtype=torch.long),
        }
