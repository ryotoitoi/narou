import datetime
import os
import shutil

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split

from wtfml.data_loaders.nlp.classification import BERTSimpleDataset
from wtfml.data_loaders.pl_data_module.data_module import plDataModule
from wtfml.engine.nlp.model import BERTBaseClassifier
from wtfml.engine.pl_engine.BERT_classification import BERTClassificationPlEngine

wandb.init(project="narou", entity="ryotoitoi")
wandb_logger = WandbLogger(project="narou")

d_today = datetime.date.today()

MAX_EPOCH = 20
save_folder = "exp_1/model/{}".format(d_today)
train_data_path = "data/train.csv"


data_df = pd.read_csv(train_data_path)

sentence_list = []
cnt = 0
for t, s, k in zip(data_df["title"], data_df["story"], data_df["keyword"]):
    if t is np.nan:
        sentence = s+k
        sentence_list.append(sentence)
    elif k is np.nan:
        sentence = t+s    
        sentence_list.append(sentence)
    else:
        sentence = t+s+k
        sentence_list.append(sentence)
data_df["sentence"] = sentence_list

input_train , input_val = train_test_split(data_df, test_size=0.2, shuffle=True)
input_train = input_train.reset_index()
input_val = input_val.reset_index()
target_train = input_train["fav_novel_cnt_bin"]
target_val = input_val["fav_novel_cnt_bin"]


train_dataset = BERTSimpleDataset(
    input_texts=input_train["sentence"], target=target_train
)
val_dataset = BERTSimpleDataset(
    input_texts=input_val["sentence"], target=target_val
)

data_module = plDataModule(
    train_dataset=train_dataset, val_dataset=val_dataset, train_batch_size=16
)

classification_model = BERTBaseClassifier(num_classes=5)

pl_engine = BERTClassificationPlEngine(
    model=classification_model,
    lr=1e-5,
    max_epoch=MAX_EPOCH,
)

callbacks_path = save_folder

if not os.path.exists(callbacks_path):
    os.makedirs(callbacks_path)
input_val.to_csv(
    os.path.join(callbacks_path, "valid_table.csv")
)  # 度のデータをvalidationに利用したのかの記録

#
callbacks_loss = pl.callbacks.ModelCheckpoint(
    dirpath=callbacks_path,
    filename="{epoch}-{valid_loss:.4f}-{valid_acc:.4f}",
    monitor="valid_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)

tb_logger = pl_loggers.TensorBoardLogger(os.path.join(save_folder, "logs/"))

early_stopping = EarlyStopping(
monitor = "valid_loss",
mode = "min",
patience=3
)

trainer = pl.Trainer(
    gpus = 1,
    max_epochs=MAX_EPOCH,
    gradient_clip_val=0.5,
    logger=[tb_logger, wandb_logger],
    callbacks=[callbacks_loss, early_stopping],
)
trainer.fit(pl_engine, datamodule=data_module)

# memory leakingの対策
pl_engine.model.cpu()
for optimizer_metrics in trainer.optimizers[0].state.values():
    for metric_name, metric in optimizer_metrics.items():
        if torch.is_tensor(metric):
            optimizer_metrics[metric_name] = metric.cpu()