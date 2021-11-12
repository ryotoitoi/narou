# lightgbmを試す。

import pandas as pd
import numpy as np
import re
from glob import glob
from tqdm import tqdm
import optuna
import umap
import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.model_selection import train_test_split
from wandb.lightgbm import wandb_callback
import wandb
from sklearn.model_selection import RepeatedKFold

wandb.init(project="narou", entity="ryotoitoi")

### ファイル読み込み・データ確認

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
sub_df = pd.read_csv('./data/sample_submission.csv')

df_train_num = df_train.select_dtypes("int")
df_test_num = df_test.select_dtypes("int")

train_title = np.load("./npy/train_title_roberta.npy")
train_story = np.load("./npy/train_story_roberta.npy")
train_key = np.load("./npy/train_keyword_roberta.npy")
test_title = np.load("./npy/test_title_roberta.npy")
test_story = np.load("./npy/test_story_roberta.npy")
test_key = np.load("./npy/test_keyword_roberta.npy")

print(train_title.shape)
print(train_story.shape)
print(train_key.shape)
print(test_title.shape)
print(test_story.shape)
print(test_key.shape)
key = np.concatenate([train_key, test_key])
title = np.concatenate([train_title, test_title])
story = np.concatenate([train_story, test_story])
print(key.shape)
print(title.shape)
print(story.shape)


um = umap.UMAP(random_state=42)
um.fit(key)
train_key_emb = um.fit_transform(train_key)
test_key_emb = um.fit_transform(test_key)

um = umap.UMAP(random_state=42)
um.fit(title)
train_title_emb = um.fit_transform(train_title)
test_title_emb = um.fit_transform(test_title)

um = umap.UMAP(random_state=42)
um.fit(story)
train_story_emb = um.fit_transform(train_story)
test_story_emb = um.fit_transform(test_story)
train_key_df = pd.DataFrame(train_key_emb).rename(columns={0:"key_0", 1:"key_1"})
train_title_df = pd.DataFrame(train_title_emb).rename(columns={0:"title_0", 1:"title_1"})
train_story_df = pd.DataFrame(train_story_emb).rename(columns={0:"story_0", 1:"story_1"})
test_key_df = pd.DataFrame(test_key_emb).rename(columns={0:"key_0", 1:"key_1"})
test_title_df = pd.DataFrame(test_title_emb).rename(columns={0:"title_0", 1:"title_1"})
test_story_df = pd.DataFrame(test_story_emb).rename(columns={0:"story_0", 1:"story_1"})

df_train = pd.concat([df_train_num, train_key_df, train_title_df, train_story_df], axis=1)
df_test = pd.concat([df_test_num, test_key_df, test_title_df, test_story_df], axis=1)
print(df_train.shape)
print(df_test.shape)

train_x = df_train.drop(columns="fav_novel_cnt_bin")
train_y = df_train[["fav_novel_cnt_bin"]]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
params = {
    'objective': 'multiclass',
    'num_classes': 5,
    "verbosity": -1,
    'metric': 'multi_logloss',
    "seed": 42
}

train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(val_x, label=val_y)

cat_cols = ['userid', 'biggenre', 'genre', 'novel_type', 'end', 'isstop', 'isr15', 'isbl', 'isgl', 'iszankoku', 'istensei', 'istenni', 'pc_or_k']


model = lgb.train(
    params,
    train_data, 
    categorical_feature = cat_cols,
    valid_names = ['train', 'valid'],
    valid_sets =[train_data, val_data], 
    verbose_eval = 50,
    callbacks=[wandb_callback(), early_stopping(50), log_evaluation(50)], 
)

val_pred = model.predict(val_x, num_iteration=model.best_iteration)

pred_df = pd.DataFrame(sorted(zip(val_x.index, val_pred, val_y)), columns=['index', 'predict', 'actual'])

feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(), train_x.columns)), columns=['importance', 'feature'])

test_pred = model.predict(df_test, num_iteration=model.best_iteration)
sub_df.iloc[:, 1:] = test_pred
sub_df.to_csv('./output/lgb_emb_test_submission.csv', index=False)