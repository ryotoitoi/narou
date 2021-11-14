# lightgbmを試す。
import pandas as pd
import numpy as np
import re
from glob import glob
from tqdm import tqdm
import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from tqdm.notebook import tqdm
import sys 
tqdm.pandas()
## dataのロード
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
sub_df = pd.read_csv('../data/sample_submission.csv')
## ncodeを数値に置き換える
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
import numpy as np
import pandas as pd
print("download .npy file")
train_title = np.load("../npy/train_title_roberta.npy")
train_story = np.load("../npy/train_story_roberta.npy")

test_title = np.load("../npy/test_title_roberta.npy")
test_story = np.load("../npy/test_story_roberta.npy")
## RoBERTaでベクトル化したやつを主成分分析をする
# 行列の標準化
title = np.concatenate([train_title, test_title])
story = np.concatenate([train_story, test_story])

title = pd.DataFrame(title).progress_apply(lambda x: (x-x.mean())/x.std(), axis=0)
story = pd.DataFrame(story).progress_apply(lambda x: (x-x.mean())/x.std(), axis=0)

train_title = title[:40000]
train_story = story[:40000]

test_title = title[40000:]
test_story = story[40000:]

train_title_df = pd.DataFrame(train_title)
train_story_df = pd.DataFrame(train_story)
test_title_df = pd.DataFrame(test_title)
test_story_df = pd.DataFrame(test_story)

for col_name in train_title_df.columns:
    train_title_df = train_title_df.rename(columns = {col_name:f"title_{col_name}"})
for col_name in train_story_df.columns:
    train_story_df = train_story_df.rename(columns = {col_name:f"story_{col_name}"})
for col_name in test_title_df.columns:
    test_title_df = test_title_df.rename(columns = {col_name:f"title_{col_name}"})
for col_name in test_story_df.columns:
    test_story_df = test_story_df.rename(columns = {col_name:f"story_{col_name}"})

## Universal Sentence Encoderのロード
train_title_univ = np.load("../npy/train_title_universal.npy")
test_title_univ = np.load("../npy/test_title_universal.npy")
title_univ = np.concatenate([train_title_univ, test_title_univ])
title_univ = pd.DataFrame(title_univ).progress_apply(lambda x: (x-x.mean())/x.std(), axis=0)
train_title_univ = title_univ[:40000]
test_title_univ = title_univ[40000:]
train_title_univ_df = pd.DataFrame(train_title_univ)
test_title_univ_df = pd.DataFrame(test_title_univ)
## dfをまとめる
df_train = pd.concat([df_train_num, train_title_df, train_title_univ_df, train_story_df, df_train[["general_firstup"]]], axis=1)
df_test = pd.concat([df_test_num, test_title_df, test_title_univ_df, test_story_df], axis=1)
## 学習データの期間を変更してみる
df_train["datetime"] = df_train['general_firstup'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
df_train = df_train[df_train["datetime"] > datetime.date(2020,1,1)].drop(columns=["datetime", "general_firstup"])
print(df_train.shape)
## 作成したデータを保存する
import os
os.makedirs("./data", exist_ok=True)

print(df_train.shape)
print(df_test.shape)

df_train.to_pickle("./data/train.pkl")
df_test.to_pickle("./data/test.pkl")
