"""
__author__: Abhishek Thakur
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from wtfml.engine.nlp.model import BERTBaseClassifier


class BertMTClassificationPlEngine(pl.LightningModule):
    """
    BERTでマルチタスクラーニングをする
    """

    def __init__(
        self,
        model=BERTBaseClassifier(num_classes=2 + 8),
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
        output_attentions: bool = False,
    ):
        """
        Args:
            model ([type], optional): [重要度（2値）＋カテゴリ数（8クラス）]. Defaults to BERTBaseClassifier(num_classes=2 + 8).
            importance_loss_fn ([type], optional): [重要度の損失関数]. Defaults to nn.CrossEntropyLoss().
            category_loss_fn ([type], optional): [カテゴリの損失関数]. Defaults to nn.CrossEntropyLoss(ignore_index=0).
            train_importance_precision_4 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision( num_classes=2, threshold=0.4 ).
            train_importance_precision_9 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision( num_classes=2, threshold=0.9 ).
            train_category_accuracy ([type], optional): [訓練時のカテゴリの正解率を計算する]. Defaults to torchmetrics.Accuracy().
            val_importance_precision_4 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision(num_classes=2, threshold=0.4).
            val_importance_precision_9 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision(num_classes=2, threshold=0.9).
            val_category_accuracy ([type], optional): [検証時のカテゴリの正解率を計算する]. Defaults to torchmetrics.Accuracy().
            lambda_category (float, optional): [マルチタスクラーニングをする際のロスの重み]. Defaults to 0.5.
            lr (float, optional): [学習率]. Defaults to 3e-5.
            max_epoch (int, optional): [最大エポック数]. Defaults to 10.
        """
        super(BertMTClassificationPlEngine, self).__init__()
        self.model = model
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
        self.output_attentions = output_attentions

    def forward(self, ids, mask, token_type_ids):
        x, attention = self.model(ids, mask, token_type_ids)
        if self.output_attentions:
            return x, attention[-1]
        else:
            return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        ids, mask, token_type_ids, is_important, category = (
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
            batch["is_important"],
            batch["category"],
        )

        pred_batch_train = self.forward(ids, mask, token_type_ids)

        pred_is_important = pred_batch_train[:, :2]
        pred_category = pred_batch_train[:, 2:]

        importance_loss = self.importance_loss_fn(
            pred_is_important, nn.Flatten(start_dim=0)(is_important)
        )
        category_loss = self.category_loss_fn(pred_category, category)
        train_loss = importance_loss + self.lambda_category * category_loss

        pred_batch_train_for_metrics = torch.sigmoid(pred_batch_train)
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
            "train_importance_loss_4",
            importance_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_category_loss",
            category_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "train_importance_precision_theshold_4",
            self.train_importance_precision_4,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_importance_precision_theshold_9",
            self.train_importance_precision_9,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_category_accuracy",
            self.train_category_accuracy,
            on_step=True,
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
        ids, mask, token_type_ids, is_important, category = (
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
            batch["is_important"],
            batch["category"],
        )

        out = self.forward(ids, mask, token_type_ids)
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

        self.log(
            "valid_loss",
            val_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {
            "val_loss": val_loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.lr)

        # opt = optim.AdamW(self.model.parameters(), lr=self.lr)
        sch = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=5, num_training_steps=self.max_epoch
        )
        return [opt], [sch]


class DistilBertMTClassificationPlEngine(pl.LightningModule):
    """
    BERTでマルチタスクラーニングをする
    """

    def __init__(
        self,
        model=BERTBaseClassifier(num_classes=2 + 8),
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
        output_attentions: bool = False,
    ):
        """
        Args:
            model ([type], optional): [重要度（2値）＋カテゴリ数（8クラス）]. Defaults to BERTBaseClassifier(num_classes=2 + 8).
            importance_loss_fn ([type], optional): [重要度の損失関数]. Defaults to nn.CrossEntropyLoss().
            category_loss_fn ([type], optional): [カテゴリの損失関数]. Defaults to nn.CrossEntropyLoss(ignore_index=0).
            train_importance_precision_4 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision( num_classes=2, threshold=0.4 ).
            train_importance_precision_9 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision( num_classes=2, threshold=0.9 ).
            train_category_accuracy ([type], optional): [訓練時のカテゴリの正解率を計算する]. Defaults to torchmetrics.Accuracy().
            val_importance_precision_4 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision(num_classes=2, threshold=0.4).
            val_importance_precision_9 ([type], optional): [precisionを計算]. Defaults to torchmetrics.Precision(num_classes=2, threshold=0.9).
            val_category_accuracy ([type], optional): [検証時のカテゴリの正解率を計算する]. Defaults to torchmetrics.Accuracy().
            lambda_category (float, optional): [マルチタスクラーニングをする際のロスの重み]. Defaults to 0.5.
            lr (float, optional): [学習率]. Defaults to 3e-5.
            max_epoch (int, optional): [最大エポック数]. Defaults to 10.
        """
        super(DistilBertMTClassificationPlEngine, self).__init__()
        self.model = model
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
        self.output_attentions = output_attentions

    def forward(self, ids, mask,):
        x, attention = self.model(ids, mask,)
        if self.output_attentions:
            return x, attention[-1]
        else:
            return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        ids, mask, is_important, category = (
            batch["ids"],
            batch["mask"],

            batch["is_important"],
            batch["category"],
        )

        pred_batch_train = self.forward(ids, mask, )

        pred_is_important = pred_batch_train[:, :2]
        pred_category = pred_batch_train[:, 2:]

        importance_loss = self.importance_loss_fn(
            pred_is_important, nn.Flatten(start_dim=0)(is_important)
        )
        category_loss = self.category_loss_fn(pred_category, category)
        train_loss = importance_loss + self.lambda_category * category_loss

        pred_batch_train_for_metrics = torch.sigmoid(pred_batch_train)
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
            "train_importance_loss_4",
            importance_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_category_loss",
            category_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "train_importance_precision_theshold_4",
            self.train_importance_precision_4,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_importance_precision_theshold_9",
            self.train_importance_precision_9,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "train_category_accuracy",
            self.train_category_accuracy,
            on_step=True,
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
        ids, mask,  is_important, category = (
            batch["ids"],
            batch["mask"],

            batch["is_important"],
            batch["category"],
        )

        out = self.forward(ids, mask,)
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

        self.log(
            "valid_loss",
            val_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {
            "val_loss": val_loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.lr)

        # opt = optim.AdamW(self.model.parameters(), lr=self.lr)
        sch = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=5, num_training_steps=self.max_epoch
        )
        return [opt], [sch]
