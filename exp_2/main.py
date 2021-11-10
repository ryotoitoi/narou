import pandas as pd
import numpy as np
import re
from glob import glob
from tqdm import tqdm
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import train_test_split
from wandb.lightgbm import wandb_callback
import wandb
from utils.cross_validation import FoldGenerator


wandb.init(project="narou", entity="ryotoitoi")

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
sub_df = pd.read_csv('./data/sample_submission.csv')

df_train = df_train.select_dtypes("int")
df_test = df_test.select_dtypes("int")

train_x = df_train.drop(columns="fav_novel_cnt_bin")
train_y = df_train[["fav_novel_cnt_bin"]]


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)

print(train_x.head())
print(train_y.head())

SEED = 0

params = {
    'objective': 'multiclass',
    'num_classes': 5,
    'metric': 'multi_logloss',
}


train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(val_x, label=val_y)

cat_cols = ['userid', 'biggenre', 'genre', 'novel_type', 'end', 'isstop', 'isr15', 'isbl', 'isgl', 'iszankoku', 'istensei', 'istenni', 'pc_or_k']
best_params, tuning_history = dict(), list()
model = lgb.train(
    params,
    train_data, 
    categorical_feature = cat_cols,
    valid_names = ['train', 'valid'],
    valid_sets =[train_data, val_data], 
    verbose_eval = 100,
    callbacks=[wandb_callback()], 
)
model = lgb.train(
    model.params,
    train_data, 
    categorical_feature = cat_cols,
    valid_names = ['train', 'valid'],
    valid_sets =[train_data, val_data], 
    verbose_eval = 100,
    callbacks=[wandb_callback()], 
)
val_pred = model.predict(val_x, num_iteration=model.best_iteration)

pred_df = pd.DataFrame(sorted(zip(val_x.index, val_pred, val_y)), columns=['index', 'predict', 'actual'])

feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(), train_x.columns)), columns=['importance', 'feature'])

test_pred = model.predict(df_test, num_iteration=model.best_iteration)
sub_df.iloc[:, 1:] = test_pred
sub_df.to_csv('./output/test_submission.csv', index=False)