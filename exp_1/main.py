from typing import Optional, Tuple, Union
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import hydra
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchmetrics.functional import auroc


# ここからmodel_train
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )

    # callbacks_path = os.path.join(
    #     wandb_logger.experiment.dir, cfg.path.checkpoint_path
    # )

    # callbacks_path = os.path.join(save_folder, "{}".format(fold))
    wandb_logger.log_hyperparams(cfg)
    
    d_today = datetime.date.today()


    save_folder = "model/{}".format(d_today)



    data_df = pd.read_csv(cfg.path.data_file_path).dropna(subset=["title"]).sample(10).reset_index(drop = True)
    target = data_df["judgement"]
    input_data = data_df["title"]

    fold_generator = FoldGenerator(
        targets=target,
        task="binary_classification",
        num_splits=cfg.training.num_split,
        shuffle=True,
    )

    for fold in range(cfg.training.num_split):
        (
            _,
            _,
            input_train,
            input_val,
            target_train,
            target_val,
        ) = fold_generator.get_fold(data=data_df, fold=fold)

        # Samplerの定義
        target = np.array(input_train["judgement"].astype(int))
        print('target train 0/1: {}/{}'.format(
            len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        weight_random_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

        # DataSetの定義
        train_dataset = BERTSimpleDataset(
            input_texts=input_train["title"], target=target_train, 
        )
        val_dataset = BERTSimpleDataset(
            input_texts=input_val["title"], target=target_val
        )

        data_module = plDataModule(
            train_dataset=train_dataset, val_dataset=val_dataset, 
            train_batch_size=cfg.training.batch_size, 
            val_batch_size = cfg.training.batch_size, 
            train_sampler =  weight_random_sampler
        )

        classification_model = BERTBaseClassifier(
            num_classes=cfg.model.n_classes, 
            pretrain_model_name = cfg.model.pretrained_model_name
            )

        pl_engine = BERTClassificationPlEngine(
            model=classification_model,
            lr=cfg.training.learning_rate,
            max_epoch=cfg.training.n_epochs,
        )

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        input_val.to_csv(
            os.path.join(checkpoint_path, "valid_table.csv")
        )  # 度のデータをvalidationに利用したのかの記録

        callbacks_loss = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="{epoch}-{valid_loss:.4f}-{valid_acc:.4f}",
            monitor="valid_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        early_stopping = EarlyStopping(
            monitor = "valid_loss",
            mode = "min",
            patience=cfg.callbacks.patience
            )

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=cfg.training.n_epochs,
            gradient_clip_val=0.5,
            logger=wandb_logger,
            callbacks=[callbacks_loss, early_stopping],
        )
        trainer.fit(pl_engine, datamodule=data_module)

        # memory leakingの対策
        pl_engine.model.cpu()
        for optimizer_metrics in trainer.optimizers[0].state.values():
            for metric_name, metric in optimizer_metrics.items():
                if torch.is_tensor(metric):
                    optimizer_metrics[metric_name] = metric.cpu()

if __name__ == "__main__":
    main()
    print("hello!")