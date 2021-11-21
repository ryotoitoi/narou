import pandas as pd
import numpy as np
import os
import re
from glob import glob
from tqdm import tqdm
import optuna
# import optuna.integration.lightgbm as lgb
import lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from wandb.lightgbm import wandb_callback
import wandb

from preprocessing import preprocessing

from tqdm.auto import tqdm
tqdm.pandas()
preprocessing()

exp_num = "exp7"
wandb.init(project="narou", entity="ryotoitoi", name = f"{exp_num}_narou")

### ファイル読み込み・データ確認

df_train = pd.read_pickle(f'{exp_num}/data/train.pkl')
df_test = pd.read_pickle(f'{exp_num}/data/test.pkl')
sub_df = pd.read_csv('./data/sample_submission.csv')
X = df_train.drop(columns="fav_novel_cnt_bin")
y = df_train[["fav_novel_cnt_bin"]]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_x, val_x = X.iloc[train_index, :], X.iloc[test_index, :]
    train_y, val_y = y.iloc[train_index, :], y.iloc[test_index, :]
    params = {
        'objective': 'multiclass',
        'num_classes': 5,
        "verbosity": -1,
        'metric': 'multi_logloss',
        "seed": 42
    }

    train_data = lgb.Dataset(train_x, label=train_y)
    val_data = lgb.Dataset(val_x, label=val_y)

    cat_cols = ["ncode_num", "userid", 'biggenre', 'genre', 'novel_type', 'end', 'isstop', 'isr15', 'isbl', 'isgl', 'iszankoku', 'istensei', 'istenni', 'pc_or_k']

    print("start training")
    model = lgb.train(
        params,
        train_data, 
        categorical_feature = cat_cols,
        valid_names = ['train', 'valid'],
        valid_sets =[train_data, val_data], 
        callbacks=[wandb_callback(), early_stopping(10), log_evaluation(10)], 
    )

    test_pred = model.predict(df_test, num_iteration=model.best_iteration)
    sub_df.iloc[:, 1:] = test_pred
    os.makedirs("{exp_num}/output", exist_ok=True)
    sub_df.to_csv(f'{exp_num}/output/{exp_num}_fold{fold}.csv', index=False)
    fold += 1

df_0 = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold0.csv")
df_1 = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold1.csv")
df_2 = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold2.csv")
df_3 = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold3.csv")
df_4 = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold4.csv")
pd.concat([df_0 , df_1, df_2, df_3, df_4]).groupby("ncode").mean().reset_index().to_csv("./output/sub.csv", index=False)