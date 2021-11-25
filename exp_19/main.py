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

exp_num = "exp_19"

# ファイル読み込み・データ確認

df_train = pd.read_pickle(f'{exp_num}/data/train.pkl')
df_test = pd.read_pickle(f'{exp_num}/data/test.pkl')
sub_df = pd.read_csv('./data/sample_submission.csv')
X = df_train.drop(columns="fav_novel_cnt_bin")
y = df_train[["fav_novel_cnt_bin"]]

GPU_ENABLED = False

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X,y, test_size=0.2)
    # カテゴリのカラムのみを抽出
    categorical_features_indices = np.where((X.dtypes != np.float32) & (X.dtypes != np.float64))[0]


    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train_pool = Pool(train_x, train_y, cat_features=categorical_features_indices)
    validate_pool = Pool(valid_x, valid_y, cat_features=categorical_features_indices)

    params = {
        'iterations' : trial.suggest_int('iterations', 50, 300),                         
        'depth' : trial.suggest_int('depth', 4, 10),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
        'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'learning_rate' :trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter'])
    }

    gbm = CatBoostClassifier(loss_function='MultiClass',task_type= "GPU", l2_leaf_reg=50, **params)

    gbm.fit(train_pool, eval_set=validate_pool, verbose=0, early_stopping_rounds=30)

    preds = gbm.predict_proba(valid_x)

    multilog_loss = log_loss(valid_y, preds)
    return multilog_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))
print("best params", trial.params)

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

    params = trial.params
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


print("Finish Training.")
sub_df = pd.DataFrame()
for i in range(10):
    tmp_df = pd.read_csv(f"./{exp_num}/output/{exp_num}_fold{i}.csv")
    sub_df = pd.concat([sub_df, tmp_df])

sub_df.groupby("ncode").mean().reset_index().to_csv(f"./{exp_num}/output/submit.csv", index=False)
