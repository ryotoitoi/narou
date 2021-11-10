import datetime
import os
import shutil

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from wtfml.data_split.cross_validation import FoldGenerator
from wtfml.data_loaders.nlp.classification import BERTSimpleDataset
from wtfml.data_loaders.pl_data_module.data_module import plDataModule
from wtfml.engine.nlp.model import BERTBaseClassifier
from wtfml.engine.pl_engine.BERT_classification import BERTClassificationPlEngine

wandb.init(project="narou", entity="ryotoitoi")
wandb_logger = WandbLogger(project="narou")

d_today = datetime.date.today()

NUM_SPLIT = 5
MAX_EPOCH = 3
save_folder = "./model/{}".format(d_today)
train_data_path = "./data/train.csv"


data_df = pd.read_csv(train_data_path).dropna(subset=["title"]).reset_index()
target = data_df["fav_novel_cnt_bin"]
input_data = data_df["title"]

fold_generator = FoldGenerator(
    targets=target,
    task="multiclass_classification",
    num_splits=NUM_SPLIT,
    shuffle=True,
    random_state = 42
)

for fold in range(NUM_SPLIT):
    (
        _,
        _,
        input_train,
        input_val,
        target_train,
        target_val,
    ) = fold_generator.get_fold(data=data_df, fold=fold)
    print(f"fold {fold}")
    train_dataset = BERTSimpleDataset(
        input_texts=input_train.reindex()["title"], target=target_train.reindex(), category=input_train.reindex()["biggenre"]
    )
    val_dataset = BERTSimpleDataset(
        input_texts=input_val.reindex()["title"], target=target_val.reindex(), category=input_val.reindex()["biggenre"]
    )

    data_module = plDataModule(
        train_dataset=train_dataset, val_dataset=val_dataset, train_batch_size=32
    )

    classification_model = BERTBaseClassifier(num_classes=5)

    pl_engine = BERTClassificationPlEngine(
        model=classification_model,
        lr=1e-5,
        max_epoch=MAX_EPOCH,
    )

    callbacks_path = os.path.join(save_folder, "{}".format(fold))

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
        gpus = 0,
        max_epochs=MAX_EPOCH,
        gradient_clip_val=0.5,
        logger=[tb_logger,wandb_logger], 
        callbacks=[callbacks_loss, early_stopping],
    )
    trainer.fit(pl_engine, datamodule=data_module)

    # memory leakingの対策
    pl_engine.model.cpu()
    for optimizer_metrics in trainer.optimizers[0].state.values():
        for metric_name, metric in optimizer_metrics.items():
            if torch.is_tensor(metric):
                optimizer_metrics[metric_name] = metric.cpu()