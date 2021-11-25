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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from catboost import Pool
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
tqdm.pandas()



exp_num = "exp_17"

# ファイル読み込み・データ確認

df_train = pd.read_pickle(f'{exp_num}/data/train.pkl')
df_test = pd.read_pickle(f'{exp_num}/data/test.pkl')
sub_df = pd.read_csv('./data/sample_submission.csv')
X = df_train.drop(columns="fav_novel_cnt_bin")
y = df_train[["fav_novel_cnt_bin"]]

GPU_ENABLED = True

print("start tuning!")
def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X,y, test_size=0.2)
    # カテゴリのカラムのみを抽出
    categorical_features_indices = np.where((X.dtypes != np.float32) & (X.dtypes != np.float64))[0]


    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train_pool = Pool(train_x, train_y, cat_features=categorical_features_indices)
    validate_pool = Pool(valid_x, valid_y, cat_features=categorical_features_indices)

    param = {
        'loss_function': 'MultiClass',
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.1),
        "iterations": trial.suggest_int("iterations", 1000, 5000),
        "depth": trial.suggest_int("depth", 3, 12),
        'random_strength': trial.suggest_uniform('random_strength',10,50),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        'custom_loss': ['Recall', "Precision", "F1"],
        'random_seed': 42,
        "verbose": True,
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 100)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    
    if GPU_ENABLED:
        params["task_type"] = "GPU"

    gbm = CatBoostClassifier(**param)

    gbm.fit(train_pool, eval_set=validate_pool, verbose=0, early_stopping_rounds=30)

    preds = gbm.predict_proba(valid_x)

    multilog_loss = log_loss(valid_y, preds)
    return multilog_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))
print("finish optuna!")

# CVを開始
print("start cv")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 0
print("start training")
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_x, val_x = X.iloc[train_index, :], X.iloc[test_index, :]
    train_y, val_y = y.iloc[train_index, :], y.iloc[test_index, :]

    # カテゴリのカラムのみを抽出
    categorical_features_indices = np.where((train_x.dtypes != np.float32) & (X.dtypes != np.float64))[0]

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
        'learning_rate': 0.02,       # 学習率
        'early_stopping_rounds': 30,
        'iterations': 10000,
        'custom_loss': ['Accuracy'],
        'random_seed': 42,
        "verbose": True,
        'task_type': "GPU",
    }
    # パラメータを指定した場合は、以下のようにインスタンスに適用させる
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validate_pool)

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
fig.write_html("feature_importance.html")

print("Finish Training.")
sub_df = pd.DataFrame()
for i in range(10):
    tmp_df = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold{i}.csv")
    sub_df = pd.concat([sub_df, tmp_df])

sub_df.groupby("ncode").mean().reset_index().to_csv(f"./{exp_num}/output/submit.csv", index=False)
