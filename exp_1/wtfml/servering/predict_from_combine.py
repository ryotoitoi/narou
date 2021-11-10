#%%
from typing import Optional
import albumentations as A
import cv2
import numpy as np
import onnxruntime
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    BertJapaneseTokenizer,
)
from wtfml.data_loaders.nlp.utils import clean_sentence
from wtfml.utils.utils import np_softmax

#%%


class CombinePredictor:
    def __init__(
        self,
        model_path: str,
        device="cpu",
        tokenizer_name="cl-tohoku/bert-base-japanese-whole-word-masking",
        clearning_function=clean_sentence,
    ):

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_name)
        self.clearning_function = clearning_function
        self.image_pretreatment = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        )
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:  # TODO GPUのときのプロバイダーの実装
            raise ValueError(
                "GPUのONNXランタイムの実装はまだです。。。(実装していただければご報告ください。テンプレートを書き換えます。)"
            )
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers
        )
        self.device = device

    def predict(self, tweet_text: str, image: Optional[np.ndarray] = None, image_path:Optional[str] = None):
        if self.clearning_function:
            tweet_text = self.clearning_function(tweet_text)

        inputs = self.tokenizer.encode_plus(
            tweet_text,
            None,
            add_special_tokens=True,
            truncation=True,
        )

        self.token = [
            token.replace("#", "")
            for token in self.tokenizer.convert_ids_to_tokens(inputs["input_ids"])[1:-1]
        ]
        if not image and image_path:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_CUBIC,
        )
        image = self.image_pretreatment(image=image)["image"]
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))

        ort_inputs = {
            "ids": np.expand_dims(inputs["input_ids"], 0),
            "mask": np.expand_dims(inputs["attention_mask"], 0),
            "token_type_ids": np.expand_dims(inputs["token_type_ids"], 0),
            "image": image,
        }

        prediction, attention_raw, image_attention = self.session.run(None, ort_inputs)
        is_importance_prediction = np_softmax(prediction[:, :2])[0]
        category_prediction = np_softmax(prediction[:, 2:])[0]

        seq_len = attention_raw.shape[2]
        all_attens = np.zeros(seq_len)

        for i in range(12):
            all_attens += attention_raw[0, i, 0, :]

        return {
            "is_important": is_importance_prediction,
            "category_prediction": category_prediction,
            "tokens": self.token,
            "text_attention": all_attens[1:-1],
            "image_attention": image_attention,
        }


# %%
