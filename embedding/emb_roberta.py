# RoBERTによるテキストのベクトル化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import re
import regex
import torch
import transformers
import json
import os
import emoji
import mojimoji
import neologdn
from tqdm.auto import tqdm
tqdm.pandas()



emoji_json_path = "./emoji/emoji_ja.json"
json_open = open(emoji_json_path)
emoji_dict = json.load(json_open)


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"<[^>]*?>", "", sentence)  # タグ除外
    sentence = mojimoji.zen_to_han(sentence, kana=False)
    sentence = neologdn.normalize(sentence)
    sentence = re.sub(
        r'[!"#$%&\'\\\\()*+,\-./:;<=>?@\[\]\^\_\`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠？！｀＋￥％︰-＠]。、♪',
        " ",
        sentence,
    )  # 記号
    sentence = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", sentence)
    sentence = re.sub(r"[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+", " ", sentence)

    sentence = "".join(
        [
            "絵文字" + emoji_dict.get(c, {"short_name": ""}).get("short_name", "")
            if c in emoji.UNICODE_EMOJI["en"]
            else c
            for c in sentence
        ]
    )
    return sentence
class BertSequenceVectorizer:
    def __init__(self, model_name: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.do_lower_case = True 
        self.bert_model = transformers.RobertaModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 200


    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()



DATA_DIR = Path('./data')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

BSV = BertSequenceVectorizer('rinna/japanese-roberta-base')

for col in ['title', 'story', 'keyword']:
    print('##########' + col + '##########')
    train[col] = train[col].fillna('NaN')
    test[col] = test[col].fillna('NaN')
    np.save(f'./npy/train_{col}_roberta', np.stack(train[col].progress_apply(lambda x: BSV.vectorize(x))))
    np.save(f'./npy/test_{col}_roberta', np.stack(test[col].progress_apply(lambda x: BSV.vectorize(x))))