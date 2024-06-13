from torch.utils.data import Dataset
from typing import Optional

import pandas as pd
from utilsd.config import Registry


class DATASETS(metaclass=Registry, name="dataset"):
    pass


@DATASETS.register_module()
class ForecastDataset(Dataset):
    def __init__(
        self,
        feature_path: Optional[str] = None,
        label_path: Optional[str] = None,
    ):
        super().__init__()
        self.feature_path = feature_path
        self.label_path = label_path
        self.feature = None
        self.label = None

    def __len__(self):
        assert self.feature is not None and self.label is not None
        return len(self.feature)

    def __getitem__(self, index):
        assert self.feature is not None and self.label is not None
        return self.feature[index], self.label[index]

    def freeup(self):
        del self.feature
        del self.label
        del self.feature_index
        del self.label_index
        self.feature = None
        self.label = None

    def load(self):
        if self.feature is not None and self.label is not None:
            print("already loaded")
            return
        feature = pd.read_pickle(self.feature_path)
        label = pd.read_pickle(self.label_path)
        assert label.index.equals(feature.index)
        self.feature = feature.values
        self.label = label.values
        self.feature_index = feature.index
        self.label_index = label.index
        del feature
        del label

    def get_index(self):
        assert self.label is not None and self.feature is not None
        return self.label_index
