import pandas as pd
import numpy as np
import os
import re
import pickle
from glob import glob
from tqdm import tqdm
import optuna
import optuna.integration.lightgbm as lgb
# import lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.model_selection import StratifiedKFold
from wandb.lightgbm import wandb_callback
import wandb

from tqdm.auto import tqdm
tqdm.pandas()

exp_num = "exp_10"
wandb.init(project="narou", entity="ryotoitoi", name = f"{exp_num}_narou")

### ファイル読み込み・データ確認

df_train = pd.read_pickle(f'{exp_num}/data/train.pkl')
df_test = pd.read_pickle(f'{exp_num}/data/test.pkl')
sub_df = pd.read_csv('./data/sample_submission.csv')
X = df_train.drop(columns="fav_novel_cnt_bin")
y = df_train[["fav_novel_cnt_bin"]]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 0
print("start training")
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
    # 学習したモデルを保存する
    os.makedirs(f"{exp_num}/model", exist_ok=True)
    model_save_path = f'{exp_num}/model/model_fold{fold}.pkl'
    pickle.dump(model, open(model_save_path, 'wb'))

    test_pred = model.predict(df_test, num_iteration=model.best_iteration)
    sub_df.iloc[:, 1:] = test_pred
    os.makedirs(f"{exp_num}/output", exist_ok=True)
    sub_df.to_csv(f'{exp_num}/output/{exp_num}_fold{fold}.csv', index=False)
    fold += 1

    # 学習済みモデルを削除（メモリの節約的な意味）
    del model

print("Finish Training.")
sub_df = pd.DataFrame()
for i in range(10):
    tmp_df = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold{i}.csv")
    sub_df = pd.concat([sub_df, tmp_df])

sub_df.groupby("ncode").mean().reset_index().to_csv(f"./{exp_num}/output/submit.csv", index=False)