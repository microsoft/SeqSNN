"""
The dataloader for loading .ts data.
Refer to the code in https://github.com/gzerveas/mvts_transformer.
"""
from typing import Optional
import logging

import numpy as np
import pandas as pd

from sktime import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .tsforecast import DATASETS
from .utils import load_from_tsfile_to_dataframe

logger = logging.getLogger(__name__)


class Normalizer():
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean(axis=0)
                self.std = df.std(axis=0)
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )
        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


@DATASETS.register_module("ts")
class TSDataset(Dataset):
    """
    A dataset for time series forecasting.
    """

    def __init__(
        self,
        prefix: str = "data/Multivariate_ts",
        name: str = "Heartbeat",
        max_seq_len: int = 0,
        dataset: str = "train",
        dataset_split_ratio: float = 0.0,
        task: str = "classification",
        normalizer: Optional[Normalizer] = None,
    ):
        super().__init__()

        if task == "classification":
            load_data = datasets.load_from_tsfile_to_dataframe
        else:
            load_data = load_from_tsfile_to_dataframe

        if max_seq_len == 0:
            train_df, _ = load_data(
                f"{prefix}/{name}/{name}_TRAIN.ts",
            )
            train_lengths = train_df.applymap(len).values
            test_df, _ = load_data(
                f"{prefix}/{name}/{name}_TEST.ts",
            )
            test_lengths = test_df.applymap(len).values
            self._max_seq_len = int(max(np.max(train_lengths), np.max(test_lengths)))
        else:
            self._max_seq_len = max_seq_len

        original_dataset = dataset
        if dataset == "valid":
            original_dataset = "train"

        df, labels = load_data(
            f"{prefix}/{name}/{name}_{original_dataset.upper()}.ts",
        )

        if task == "classification":
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int64)
        elif task == "regression":
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        else:
            raise ValueError(f"Unknown task: {task}")

        assert len(df) == len(labels)
        self.datasize = len(df)
        lengths = df.applymap(len).values

        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(np.max(lengths[row, :]) * [row]))
                for row in range(df.shape[0])
            ),
            axis=0,
        )

        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        feature = []
        for i in range(self.datasize):
            nowdf = df.loc[i]
            _seq_len = len(nowdf)
            newdf = pd.DataFrame(
                index=[i for num in range(self._max_seq_len - _seq_len)],
                columns=nowdf.columns,
            ).fillna(0)
            feature.append(pd.concat([newdf, nowdf]))

        self.feature = pd.concat(feature)
        self.label = labels_df

        if original_dataset == "train":
            if dataset_split_ratio > 0:
                data_index = np.arange(self.datasize)
                label_np = self.label.values
                if task == "classification":
                    ind_x_train, ind_x_valid, _, _ = train_test_split(
                        data_index,
                        label_np,
                        test_size=dataset_split_ratio,
                        stratify=label_np,
                        shuffle=True,
                        random_state=42,
                    )
                elif task == "regression":
                    ind_x_train, ind_x_valid, _, _ = train_test_split(
                        data_index,
                        label_np,
                        test_size=dataset_split_ratio,
                        shuffle=True,
                        random_state=42,
                    )
                if dataset == "train":
                    self.datasize = len(ind_x_train)
                    self.feature = self.feature.loc[ind_x_train]
                    self.label = self.label.loc[ind_x_train]
                    rename_dict = {v: k for k, v in enumerate(ind_x_train)}
                    self.feature = self.feature.rename(index=rename_dict)
                    self.label = self.label.rename(index=rename_dict)
                elif dataset == "valid":
                    self.datasize = len(ind_x_valid)
                    self.feature = self.feature.loc[ind_x_valid]
                    self.label = self.label.loc[ind_x_valid]
                    rename_dict = {v: k for k, v in enumerate(ind_x_valid)}
                    self.feature = self.feature.rename(index=rename_dict)
                    self.label = self.label.rename(index=rename_dict)
            elif dataset_split_ratio == 0:
                if dataset == "valid":
                    raise ValueError("valid dataset must have dataset_split_ratio")

        if normalizer is not None:
            self.feature = normalizer.normalize(self.feature)
        else:
            normalizer = Normalizer(norm_type="standardization")
            self.feature = normalizer.normalize(self.feature)
        self.normalizer = normalizer
        self.feature = self.feature.astype(np.float32)

    def freeup(self):
        pass

    def load(self):
        pass

    @property
    def num_classes(self):
        try:
            return len(self.class_names)
        except AttributeError:
            return 1

    @property
    def num_variables(self):
        return self.feature.shape[1]

    @property
    def max_seq_len(self):
        return self._max_seq_len

    def get_normalizer(self):
        return self.normalizer

    def get_index(self):
        return self.label.index

    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        return self.feature.loc[idx].to_numpy(), self.label.loc[idx].to_numpy()

    def get_values(self):
        return self.feature.to_numpy(), self.label.to_numpy()
