"""
__author__: Abhishek Thakur
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class FastAlartFinalClassificationPlEngine(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        image_embedding_model,
        metadata_embedding_model,
        importance_loss_fn=nn.CrossEntropyLoss(),
        category_loss_fn=nn.CrossEntropyLoss(ignore_index=0),
        train_importance_precision_4=torchmetrics.Precision(
            num_classes=2, threshold=0.4
        ),
        train_importance_precision_9=torchmetrics.Precision(
            num_classes=2, threshold=0.9
        ),
        train_category_accuracy=torchmetrics.Accuracy(),
        val_importance_precision_4=torchmetrics.Precision(num_classes=2, threshold=0.4),
        val_importance_precision_9=torchmetrics.Precision(num_classes=2, threshold=0.9),
        val_category_accuracy=torchmetrics.Accuracy(),
        lambda_category=0.5,
        lr: float = 3e-5,
        max_epoch=10,
        num_classes=10,
        unfreeze_epoch=1,
        have_image: bool = True,
    ):
        super(FastAlartFinalClassificationPlEngine, self).__init__()
        self.bert_model = bert_model
        self.image_embedding_model = image_embedding_model
        self.metadata_embedding_model = metadata_embedding_model
        self.scaler = None
        self.importance_loss_fn = importance_loss_fn
        self.category_loss_fn = category_loss_fn
        self.train_importance_precision_4 = train_importance_precision_4
        self.train_importance_precision_9 = train_importance_precision_9
        self.train_category_accuracy = train_category_accuracy
        self.val_importance_precision_4 = val_importance_precision_4
        self.val_importance_precision_9 = val_importance_precision_9
        self.val_category_accuracy = val_category_accuracy
        self.lambda_category = lambda_category
        self.lr = lr
        self.max_epoch = max_epoch
        self.fc_text_only = nn.Linear(768 + 15, num_classes)  # TODO ここの数は怪しい
        self.fc_text_image = nn.Linear(768 + 1280 + 15, num_classes)
        self.unfreeze_epoch = unfreeze_epoch
        self.have_image = have_image

    def forward(
        self,
        ids,
        mask,
        token_type_ids,
        image,
        metadata,
    ):
        if self.current_epoch < self.unfreeze_epoch:  # 最終層だけ学習
            self.bert_model.eval()
            with torch.no_grad():
                bert_features = self.bert_model.get_features(ids, mask, token_type_ids)
            if self.have_image:
                self.image_embedding_model.eval()
                with torch.no_grad():
                    image_features = self.image_embedding_model.get_features(image)
        else:  # すべての層を学習
            self.bert_model.train()
            bert_features = self.bert_model.get_features(ids, mask, token_type_ids)
            if self.have_image:
                self.image_embedding_model.train()
                image_features = self.image_embedding_model.get_features(image)

        metadata_features = self.metadata_embedding_model(metadata)
        if self.have_image:
            return self.fc_text_image(
                torch.cat([bert_features, image_features, metadata_features], dim=1)
            )
        else:
            return self.fc_text_only(
                torch.cat([bert_features, metadata_features], dim=1)
            )

    def training_step(self, batch, batch_idx):
        # REQUIRED
        (
            ids,
            mask,
            token_type_ids,
            is_important,
            category,
            image,
            metadata,
        ) = (  # FIXME　ここ直して！
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
            batch["is_important"],
            batch["category"],
            batch["image"],
            batch["metadata"],
        )

        pred_batch_train = self.forward(
            ids,
            mask,
            token_type_ids,
            image,
            metadata,
        )

        pred_is_important = pred_batch_train[:, :2]
        pred_category = pred_batch_train[:, 2:]

        importance_loss = self.importance_loss_fn(
            pred_is_important, nn.Flatten(start_dim=0)(is_important)
        )
        category_loss = self.category_loss_fn(pred_category, category)
        train_loss = importance_loss + self.lambda_category * category_loss

        is_important = is_important.to(torch.int)
        one_hot_category = nn.functional.one_hot(category, num_classes=8)
        one_hot_pred_category = nn.functional.one_hot(
            torch.argmax(pred_category, axis=1), num_classes=8
        )

        self.train_importance_precision_4(
            nn.Softmax(dim=1)(pred_is_important), is_important
        )  # (*)softmaxいる
        self.train_importance_precision_9(
            nn.Softmax(dim=1)(pred_is_important), is_important
        )  # (*)softmaxいる
        self.train_category_accuracy(one_hot_pred_category, one_hot_category)

        self.log(
            "train_importance_precision_theshold_4",  #!この辺の名前確認(石山くんのブランチなので)
            self.train_importance_precision_4,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_importance_precision_theshold_9",
            self.train_importance_precision_9,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_category_accuracy",
            self.train_category_accuracy,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        (
            ids,
            mask,
            token_type_ids,
            is_important,
            category,
            image,
            metadata,
        ) = (  # FIXME　ここ直して！
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
            batch["is_important"],
            batch["category"],
            batch["image"],
            batch["metadata"],
        )

        out = self.forward(
            ids,
            mask,
            token_type_ids,
            image,
            metadata,
        )

        out_is_important = out[:, :2]
        out_category = out[:, 2:]

        importance_loss = self.importance_loss_fn(
            out_is_important, nn.Flatten(start_dim=0)(is_important)
        )
        category_loss = self.category_loss_fn(
            out_category, nn.Flatten(start_dim=0)(category)
        )
        val_loss = torch.mean(importance_loss + self.lambda_category * category_loss)

        is_important = is_important.to(torch.int)
        one_hot_category = nn.functional.one_hot(category, num_classes=8)
        one_hot_out_category = nn.functional.one_hot(
            torch.argmax(out_category, axis=1), num_classes=8
        )

        self.val_importance_precision_4(
            nn.Softmax(dim=1)(out_is_important), is_important
        )
        self.val_importance_precision_9(
            nn.Softmax(dim=1)(out_is_important), is_important
        )
        self.val_category_accuracy(one_hot_out_category, one_hot_category)
        self.log(
            "val_importance_precision_theshold_4",
            self.val_importance_precision_4,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "val_importance_precision_theshold_9",
            self.val_importance_precision_9,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "val_category_accuracy",
            self.val_category_accuracy,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return {
            "val_loss": val_loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED

        trainable_param = (
            self.image_embedding_model.parameters()
            + self.bert_model.parameters()
            + self.bert_model.parameters()
            + self.fc.parameters()
        )
        opt = Adam(trainable_param, lr=self.lr)

        sch = CosineAnnealingLR(opt, T_max=3)
        return [opt], [sch]
