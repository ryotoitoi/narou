"""
__author__: Abhishek Thakur
modified by yongtae
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy as sp
from sklearn import model_selection


class FoldGenerator:
    """
    pd.DataFrameをn_split文だけ分割する、class
    """

    def __init__(
        self,
        targets: Union[pd.DataFrame, pd.Series],
        num_splits: int = 5,
        shuffle: bool = True,
        task: str = "binary_classification",
        random_state: Optional[int] = None,
    ):
        """

        Args:
            targets (Union[pd.DataFrame, pd.Series]): [ターゲット変数の配列]
            num_splits (int, optional): [分割数]. Defaults to 5.
            shuffle (bool, optional): [シャッフルするか否か]. Defaults to True.
            task (str, optional): [タスクの種類]. Defaults to "binary_classification".
            random_state (Optional[int], optional): [ランダムシードの固定]. Defaults to None.

        """
        self.task = task
        self.targets = targets
        if isinstance(self.targets, pd.DataFrame) or isinstance(
            self.targets, pd.Series
        ):
            self.targets = self.targets.values

        if len(self.targets.shape) == 1:
            self.targets = self.targets.reshape((-1, 1))
        self.num_splits = num_splits

        if self.task == "binary_classification":
            self.folds = model_selection.StratifiedKFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multiclass_classification":
            self.folds = model_selection.StratifiedKFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multilabel_classification":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "single_col_regression":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multi_col_regression":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            raise Exception("Task not understood")

        self.splits = dict()
        for fold_, (trn, val) in enumerate(
            self.folds.split(self.targets, self.targets)
        ):
            self.splits[fold_] = dict()
            self.splits[fold_]["train_idx"] = trn
            self.splits[fold_]["valid_idx"] = val

    def get_fold(
        self, data: Union[pd.DataFrame, pd.Series], fold: int
    ) -> Tuple[
        dict,
        dict,
        Union[pd.DataFrame, pd.Series],
        Union[pd.DataFrame, pd.Series],
        np.ndarray,
        np.ndarray,
    ]:

        if fold >= self.num_splits or fold < 0:
            raise Exception("Invalid fold number specified")
        if isinstance(data, pd.DataFrame):
            if self.targets.shape[1] == 1 and self.task != "binary_classification":
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data.loc[self.splits[fold]["train_idx"], :],
                    data.loc[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data.loc[self.splits[fold]["train_idx"], :],
                    data.loc[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )

        elif isinstance(data, sp.sparse.coo.coo_matrix) or isinstance(
            data, sp.sparse.csc.csc_matrix
        ):
            if self.targets.shape[1] == 1 and self.task != "binary_classification":
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"]],
                    data[self.splits[fold]["valid_idx"]],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"]],
                    data[self.splits[fold]["valid_idx"]],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )
        else:
            if self.targets.shape[1] == 1 and self.task != "binary_classification":
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"], :],
                    data[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"], :],
                    data[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )


class DurationSpliter:
    """
    pd.DataFrameを期間を指定して分割するclass
    """

    def __init__(
        self,
        time_series: pd.Series,
        random_state: Optional[int] = None,
    ):
        self.time = time_series
        self.random_state = random_state

    def get_splits(
        self,
        data: Union[pd.DataFrame, pd.Series],
        division_timestamp: pd.Timestamp,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
    ) -> Tuple[
        dict,
        dict,
        Union[pd.DataFrame, pd.Series],
        Union[pd.DataFrame, pd.Series],
    ]:

        train_all_idx = self.time[self.time <= division_timestamp].index
        valid_all_idx = self.time[self.time > division_timestamp].index

        if isinstance(num_train, int):
            train_idx = (
                pd.DataFrame(index=train_all_idx)
                .sample(num_train, random_state=self.random_state)
                .index
            )
        else:
            train_idx = train_all_idx

        if isinstance(num_val, int):
            valid_idx = (
                pd.DataFrame(index=valid_all_idx)
                .sample(num_val, random_state=self.random_state)
                .index
            )
        else:
            valid_idx = valid_all_idx

        if isinstance(data, pd.DataFrame):
            return (
                train_idx,
                valid_idx,
                data.loc[train_idx, :],
                data.loc[valid_idx, :],
            )
        elif isinstance(data, pd.Series):
            return (
                train_idx,
                valid_idx,
                data.loc[train_idx],
                data.loc[valid_idx],
            )
