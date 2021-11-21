import pandas as pd
import numpy as np
import re
from glob import glob
from tqdm import tqdm
import optuna
# import optuna.integration.lightgbm as lgb
import lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.model_selection import train_test_split
from wandb.lightgbm import wandb_callback
import wandb

from preprocessing import preprocessing

from tqdm.auto import tqdm
tqdm.pandas()
preprocessing()
wandb.init(project="narou", entity="ryotoitoi", name = "exp_6_narou")

### ファイル読み込み・データ確認

df_train = pd.read_pickle('exp_6/data/train.pkl')
df_test = pd.read_pickle('exp_6/data/test.pkl')
sub_df = pd.read_csv('./data/sample_submission.csv')
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

cat_cols = ["ncode_num", "userid", 'biggenre', 'genre', 'novel_type', 'end', 'isstop', 'isr15', 'isbl', 'isgl', 'iszankoku', 'istensei', 'istenni', 'pc_or_k']

print("start training")
model = lgb.train(
    params,
    train_data, 
    categorical_feature = cat_cols,
    valid_names = ['train', 'valid'],
    valid_sets =[train_data, val_data], 
    callbacks=[wandb_callback(), early_stopping(30), log_evaluation(30)], 
)

test_pred = model.predict(df_test, num_iteration=model.best_iteration)
sub_df.iloc[:, 1:] = test_pred
sub_df.to_csv('./output/exp6_submission.csv', index=False)