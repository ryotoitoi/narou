# lightgbmを試す。
import pandas as pd
import numpy as np
import re
import regex
from glob import glob
from tqdm import tqdm
import datetime
from tqdm.notebook import tqdm
import sys 
from logging import getLogger
logger = getLogger(__name__)

tqdm.pandas()
import json
import os
import emoji
import mojimoji
import neologdn

exp_num = "exp_8"

working_direction = os.getcwd()

emoji_json_path = "./emoji/emoji_ja.json"
json_open = open(emoji_json_path)
emoji_dict = json.load(json_open)

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
sub_df = pd.read_csv('./data/sample_submission.csv')

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

for i in ["title", "story", "keyword"]:
    tmp_train = []
    for train_i in df_train[i]:
        try:
            tmp_train.append(clean_sentence(train_i))
        except:
            tmp_train.append(np.nan)
    tmp_test = []
    for test_i in df_test[i]:
        try:
            tmp_test.append(clean_sentence(test_i))
        except:
            tmp_test.append(np.nan)
    df_train[i] = tmp_train
    df_test[i] = tmp_test

# ncodeを数値に置き換える
def processing_ncode(input_df: pd.DataFrame):
    output_df = input_df.copy()
    
    num_dict = {chr(i): i-65 for i in range(65, 91)}
    def _processing(x, num_dict=num_dict):
        y = 0
        for i, c in enumerate(x[::-1]):
            num = num_dict[c]
            y += 26**i * num
        y *= 9999
        return y
    
    tmp_df = pd.DataFrame()
    tmp_df['_ncode_num'] = input_df['ncode'].map(lambda x: x[1:5]).astype(int)
    tmp_df['_ncode_chr'] = input_df['ncode'].map(lambda x: x[5:])
    tmp_df['_ncode_chr2num'] = tmp_df['_ncode_chr'].map(lambda x: _processing(x))
    
    output_df['ncode_num'] = tmp_df['_ncode_num'] + tmp_df['_ncode_chr2num']
    return output_df

df_train = processing_ncode(df_train)
df_test = processing_ncode(df_test)

df_train_num = df_train.select_dtypes("int")
df_test_num = df_test.select_dtypes("int")

print("download .npy file")
logger.info("download .npy file")
train_title = np.load("./npy/train_title_roberta.npy")
train_story = np.load("./npy/train_story_roberta.npy")
# train_keyword = np.load("./npy/train_keyword_roberta.npy")

test_title = np.load("./npy/test_title_roberta.npy")
test_story = np.load("./npy/test_story_roberta.npy")
# test_keyword = np.load("./npy/test_keyword_roberta.npy")


## RoBERTaでベクトル化したやつを主成分分析をする
# 行列の標準化
title = np.concatenate([train_title, test_title])
story = np.concatenate([train_story, test_story])
# keyword = np.concatenate([train_keyword, test_keyword])

train_title = title[:40000]
train_story = story[:40000]
# train_keyword = keyword[:40000]

test_title = title[40000:]
test_story = story[40000:]
# test_keyword = keyword[40000:]

train_title_df = pd.DataFrame(train_title)
train_story_df = pd.DataFrame(train_story)
# train_keyword_df = pd.DataFrame(train_keyword)
test_title_df = pd.DataFrame(test_title)
test_story_df = pd.DataFrame(test_story)
# test_keyword_df = pd.DataFrame(test_keyword)

for col_name in train_title_df.columns:
    train_title_df = train_title_df.rename(columns = {col_name:f"title_{col_name}"})
for col_name in train_story_df.columns:
    train_story_df = train_story_df.rename(columns = {col_name:f"story_{col_name}"})
# for col_name in train_keyword_df.columns:
#     train_keyword_df = train_keyword_df.rename(columns = {col_name:f"keyword_{col_name}"})
for col_name in test_title_df.columns:
    test_title_df = test_title_df.rename(columns = {col_name:f"title_{col_name}"})
for col_name in test_story_df.columns:
    test_story_df = test_story_df.rename(columns = {col_name:f"story_{col_name}"})
# for col_name in test_keyword_df.columns:
#     test_keyword_df = test_keyword_df.rename(columns = {col_name:f"keyword_{col_name}"})

## Universal Sentence Encoderのロード
train_title_univ = np.load("./npy/train_title_universal.npy")
test_title_univ = np.load("./npy/test_title_universal.npy")

train_title_univ_df = pd.DataFrame(train_title_univ)
test_title_univ_df = pd.DataFrame(test_title_univ)
logger.info("finish downloading .npy file!")

# 文字種ベースの特徴量を作成
def create_type_features(texts):
    """文字種ベースの特徴量を作成"""

    type_data = []

    for text in texts:
        tmp = []
        
        tmp.append(len(text))

        # 平仮名の文字数カウント
        p = re.compile('[\u3041-\u309F]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # カタカナの文字数カウント
        p = re.compile('[\u30A1-\u30FF]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # 漢字の文字数カウント
        p = regex.compile(r'\p{Script=Han}+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        type_data.append(tmp)

    colnames = ['length', 'hiragana_length', 'katakana_length', 'kanji_length']
    type_df = pd.DataFrame(type_data, columns=colnames)

    for colname in type_df.columns:
        if colname != 'length':
            type_df[colname] /= type_df['length']

    return type_df

# 文字種ベースの特徴量を作成
story_type_df = create_type_features(df_train["story"])
story_type_df.columns = ['story_' + colname for colname in story_type_df.columns]
title_type_df = create_type_features(df_train["title"])
title_type_df.columns = ['title_' + colname for colname in title_type_df.columns]

## dfをまとめる
df_train = pd.concat([df_train_num, train_title_df, train_story_df, df_train[["general_firstup"]], story_type_df, title_type_df], axis=1)
df_test = pd.concat([df_test_num, test_title_df, test_story_df, story_type_df, title_type_df], axis=1)


## 学習データの期間を変更してみる
df_train["datetime"] = df_train['general_firstup'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
df_train = df_train[df_train["datetime"] > datetime.date(2019,1,1)].drop(columns=["datetime", "general_firstup"])
print(df_train.shape)
## 作成したデータを保存する
import os
os.makedirs("./{exp_num}/data", exist_ok=True)

print(df_train.shape)
print(df_test.shape)

df_train.to_pickle("./{exp_num}/data/train.pkl")
df_test.to_pickle("./{exp_num}/data/test.pkl")
logger.info("finsh preprocessing")
