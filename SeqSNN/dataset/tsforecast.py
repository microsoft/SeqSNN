from typing import Optional

import numpy as np
import pandas as pd

from SeqSNN.dataset import DATASETS
from .utils import time_features
from torch.utils.data import Dataset


@DATASETS.register_module()
class TSForecastDataset(Dataset):
    def __init__(
        self,
        file: str,
        window: int,
        horizon: int,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        normalize: int = 2,
        last_label: bool = False,
        dataset_name: Optional[str] = None,
    ):
        self.window = window
        self.horizon = horizon
        if file.endswith(".txt"):
            self.raw_data = np.loadtxt(open(file), delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            self.raw_data = np.loadtxt(open(file), delimiter=",", skiprows=1, dtype=object)[:, 1:].astype(np.float32)
        elif file.endswith(".h5"):
            self.raw_data = pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
        self.dat = np.zeros(self.raw_data.shape, dtype=np.float32)
        self.n, self.m = self.dat.shape
        if (train_ratio + test_ratio) == 1 and dataset_name == "valid":
            dataset_name = "test"
        self.dataset_name = dataset_name
        self.last_label = last_label
        self._normalized(normalize)
        self._split(train_ratio, test_ratio, self.dataset_name)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.raw_data / np.max(self.raw_data)

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.dat[:, i] = self.raw_data[:, i] / np.max(np.abs(self.raw_data[:, i]))

    def _split(self, train_ratio, test_ratio, dataset_name):
        total_size = self.n - self.window - self.horizon + 1
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)
        valid_size = total_size - test_size - train_size
        if dataset_name == "train":
            self.length = train_size
            self.start_idx = 0
        elif dataset_name == "valid":
            self.length = valid_size
            self.start_idx = train_size
        elif dataset_name == "test":
            self.length = test_size
            self.start_idx = train_size + valid_size
        else:
            raise ValueError

    def freeup(self):
        pass

    def load(self):
        pass

    @property
    def num_variables(self):
        return self.raw_data.shape[1]

    def __len__(self):
        return self.length

    @property
    def max_seq_len(self):
        return self.window

    @property
    def num_classes(self):
        if self.last_label:
            return 1
        else:
            return self.raw_data.shape[1]

    def get_index(self):
        return np.arange(self.length)

    def __getitem__(self, index):
        index = index + self.start_idx
        X = self.dat[index : index + self.window, :]
        if self.last_label:
            y = self.raw_data[index + self.window + self.horizon - 1, -1]
        else:
            y = self.raw_data[index + self.window + self.horizon - 1, :]
        return X, y


@DATASETS.register_module()
class TSMSDataset(Dataset):
    def __init__(
        self,
        file: str,
        window: int,
        horizon: int,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        normalize: int = 2,
        last_label: bool = False,
        raw_label: bool = True,
        dataset_name: Optional[str] = None,
    ):
        self.window = window
        self.horizon = horizon
        if file.endswith(".txt"):
            self.raw_data = np.loadtxt(open(file), delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            self.raw_data = np.loadtxt(open(file), delimiter=",", skiprows=1, dtype=object)[:, 1:].astype(np.float32)
            self.dates = pd.DataFrame(
                np.loadtxt(open(file), delimiter=",", skiprows=1, dtype=object)[:, 0], columns=["date"]
            )
            self.dates["date"] = self.dates["date"].map(lambda x: pd.Timestamp(x))
            self.dates = time_features(self.dates, freq="t")
        elif file.endswith(".h5"):
            self.raw_data = pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
            self.dates = pd.DataFrame(pd.read_hdf(file).reset_index()["index"]).rename(columns={"index": "date"})
            self.dates = time_features(self.dates, freq="t")
        self.dat = np.zeros(self.raw_data.shape, dtype=np.float32)
        self.n, self.m = self.dat.shape
        if (train_ratio + test_ratio) == 1 and dataset_name == "valid":
            dataset_name = "test"
        self.dataset_name = dataset_name
        self.last_label = last_label
        self.raw_label = raw_label
        self._normalized(normalize)
        self._split(train_ratio, test_ratio, self.dataset_name)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.raw_data

        if normalize == 1:
            self.dat = self.raw_data / np.max(self.raw_data)

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.dat[:, i] = self.raw_data[:, i] / np.max(np.abs(self.raw_data[:, i]))
        
        if normalize == 3:
            self.dat = (self.raw_data - np.mean(self.raw_data))/(np.std(self.raw_data)+np.finfo(float).eps)

    def _split(self, train_ratio, test_ratio, dataset_name):
        total_size = self.n - self.window - self.horizon + 1
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)
        valid_size = total_size - test_size - train_size
        if dataset_name == "train":
            self.length = train_size
            self.start_idx = 0
        elif dataset_name == "valid":
            self.length = valid_size
            self.start_idx = train_size
        elif dataset_name == "test":
            self.length = test_size
            self.start_idx = train_size + valid_size
        else:
            raise ValueError

    def freeup(self):
        pass

    def load(self):
        pass

    @property
    def num_variables(self):
        if hasattr(self, "dates"):
            return self.dates.shape[1] + self.raw_data.shape[1]
        else:
            return self.raw_data.shape[1]

    def __len__(self):
        return self.length

    @property
    def max_seq_len(self):
        return self.window

    @property
    def num_classes(self):
        if self.last_label:
            return self.horizon
        else:
            return self.raw_data.shape[1] * self.horizon

    def get_index(self):
        return np.arange(self.length)

    def __getitem__(self, index):
        index = index + self.start_idx
        X = self.dat[index : index + self.window, :]
        # add time features
        if hasattr(self, "dates"):
            X = np.concatenate([X, self.dates[index : index + self.window]], axis=1)
        if self.raw_label:
            label_data = self.raw_data
        else:
            label_data = self.dat
        if self.last_label:
            y = label_data[index + self.window : index + self.window + self.horizon, -1]
        else:
            y = label_data[index + self.window : index + self.window + self.horizon, :].reshape(-1)
        assert len(y) == self.num_classes, (len(y), self.num_classes)
        return X.astype(np.float32), y.astype(np.float32)


@DATASETS.register_module()
class InformerDataset(TSMSDataset):
    def __init__(
        self,
        file: str,
        window: int,
        horizon: int,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        normalize: int = 2,
        last_label: bool = False,
        raw_label: bool = False,
        dataset_name: Optional[str] = None,
    ):
        self.window = window
        self.horizon = horizon
        if file.endswith(".txt"):
            self.raw_data = np.loadtxt(open(file), delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            self.raw_data = np.loadtxt(open(file), delimiter=",", skiprows=1, dtype=object)[:, 1:].astype(np.float32)
            self.dates = pd.DataFrame(
                np.loadtxt(open(file), delimiter=",", skiprows=1, dtype=object)[:, 0], columns=["date"]
            )
            self.dates["date"] = self.dates["date"].map(lambda x: pd.Timestamp(x))
            self.dates = time_features(self.dates, freq="t")
        elif file.endswith(".h5"):
            self.raw_data = pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
            self.dates = pd.DataFrame(pd.read_hdf(file).reset_index()["index"]).rename(columns={"index": "date"})
            self.dates = time_features(self.dates, 0, freq="t")
        self.dat = np.zeros(self.raw_data.shape, dtype=np.float32)
        self.n, self.m = self.dat.shape
        if (train_ratio + test_ratio) == 1 and dataset_name == "valid":
            dataset_name = "test"
        self.dataset_name = dataset_name
        self.last_label = last_label
        self.raw_label = raw_label
        self._normalized(normalize)
        self._split(train_ratio, test_ratio, self.dataset_name)

    @property
    def num_variables(self):
        return self.raw_data.shape[1]

    def __len__(self):
        return self.length

    @property
    def max_seq_len(self):
        return self.window

    @property
    def num_classes(self):
        return self.horizon

    def __getitem__(self, index):
        index = index + self.start_idx
        X = self.dat[index : index + self.window, :]
        # add time features
        X_mark = self.dates[index : index + self.window]

        # if hasattr(self, "dates"):
        #     X = np.concatenate([X, self.dates[index : index + self.window]], axis=1)
        if self.last_label:
            y = self.dat[index + self.window : index + self.window + self.horizon, -1]
        else:
            y = self.dat[index + self.window : index + self.window + self.horizon, :]
        y_mark = self.dates[index + self.window : index + self.window + self.horizon]
        assert len(y) == self.num_classes, (len(y), self.num_classes)
        return X.astype(np.float32), y.astype(np.float32), X_mark.astype(np.float32), y_mark.astype(np.float32)