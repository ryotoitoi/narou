import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import re
import pickle
from glob import glob
from tqdm import tqdm

import optuna
# import optuna.integration.lightgbm as lgb
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from catboost import Pool
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
tqdm.pandas()

exp_num = "exp_16"

# ファイル読み込み・データ確認

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

    # カテゴリのカラムのみを抽出
    categorical_features_indices = np.where(
        (train_x.dtypes != np.float32) & (X.dtypes != np.float64))[0]

    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train_pool = Pool(train_x, train_y,
                      cat_features=categorical_features_indices)
    validate_pool = Pool(
        val_x, val_y, cat_features=categorical_features_indices)

    params = {
        'loss_function': 'MultiClass',
        "classes_count": 5,
        'depth': 8,                  # 木の深さ
        'learning_rate': 0.05,       # 学習率
        'early_stopping_rounds': 10,
        'iterations': 10000,
        'custom_loss': ['MultiLogloss'],
        'random_seed': 42,
        "verbose": True,
        'task_type':"GPU",
    }
    # パラメータを指定した場合は、以下のようにインスタンスに適用させる
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validate_pool, early_stopping_rounds=100)

    # 学習したモデルを保存する
    os.makedirs(f"{exp_num}/model", exist_ok=True)
    model_save_path = f'{exp_num}/model/model_fold{fold}.pkl'
    pickle.dump(model, open(model_save_path, 'wb'))

    test_pred = model.predict_proba(df_test)
    sub_df.iloc[:, 1:] = test_pred
    os.makedirs(f"{exp_num}/output", exist_ok=True)
    sub_df.to_csv(f'{exp_num}/output/{exp_num}_fold{fold}.csv', index=False)
    fold += 1


# Feature Importance
feature_importance = model.get_feature_importance()
fig = go.Figure()
fig.add_trace(
    go.Bar(x=feature_importance, y=list(X.columns), orientation="h")
)
fig.show()
fig.write_image("feature_importance.png")

print("Finish Training.")
sub_df = pd.DataFrame()
for i in range(10):
    tmp_df = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold{i}.csv")
    sub_df = pd.concat([sub_df, tmp_df])

sub_df.groupby("ncode").mean().reset_index().to_csv(
    f"./{exp_num}/output/submit.csv", index=False)
