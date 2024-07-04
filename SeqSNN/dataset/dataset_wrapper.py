import numpy as np
import pandas as pd
from typing import Optional

from utilsd.config import Registry
import torch.nn as nn
from torch.utils.data import Dataset


class DATASETWRAPPER(metaclass=Registry, name="datasetwrapper"):
    pass


@DATASETWRAPPER.register_module("modelpool")
class MPWrapper:
    """
    A dataset for ensemble of multiple models.
    The saved model prediction values are probabilities.
    """
    def __init__(
        self,
        data_index: Optional[list] = None,
        model_pool_dir: Optional[str] = None,
        model_list: Optional[list] = None,
        loss_fn: str = None,
        split: str = 'train'
    ):
        """
        data_index: the dataset class index for verfication.
        model_pool_dir: the dir where model pool predictions are stored.
        model_list: the list of models for the ensemble.
        loss_fn: to formalize and check the prediction outputs from base models.
        split: train, valid, or test.
        """

        self.loss_fn = loss_fn
        eps = 1e-6

        model_preds = []
        for _, model_name in enumerate(model_list):
            df = pd.read_pickle(model_pool_dir + '/' + model_name + '/' + split + '_pre.pkl')
            assert np.all(data_index == df.index), \
                f"Prediction (index) of model {model_name} does not align to the original dataset."

            pred = df.values.astype(float)
            if (self.loss_fn == 'cross_entropy' and pred.shape[1] == 1) or \
                    (self.loss_fn == 'bce' and pred.shape[1] != 1):
                raise ValueError("The output dim of model prediction does not fit loss_fn.")
            elif self.loss_fn == 'cross_entropy':
                pred_sum = np.abs(np.sum(pred, axis=1) - 1)
                max_diff = np.max(pred_sum, 0)
                if max_diff > eps:
                    raise ValueError("The sum of probability values is not 1.")
            model_preds.append(pred)
        self.model_preds = np.concatenate(model_preds, axis=1)

    def wrap(self, dataset_cls):
        model_preds = self.model_preds
        old_getitem = dataset_cls.__getitem__

        def __getitem__(self, idx):
            data, target = old_getitem(idx)
            return [data, model_preds[idx]], target

        dataset_cls.__class__ = type("Wrapped_" + type(dataset_cls).__name__, (type(dataset_cls),), {"__getitem__": __getitem__})
        return dataset_cls
