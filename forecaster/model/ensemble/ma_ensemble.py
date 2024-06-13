from typing import Optional, List
from pathlib import Path
import copy
import types
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ...common.utils import __deepcopy__

from ..base import MODELS, BaseModel


def _init_scheduler(self, loader_length):
    self.scheduler = None
    total_iter = self.max_epoches * loader_length
    if self.sma_start_iter >= total_iter:
        raise ValueError("The hyperparameter \'sma_start_iter\' is greater than the total number of iterations of the training data set, so the ensemble is equal to a single base model.")


def _post_batch(self, iterations: int, epoch, train_loss, train_global_tracker, validset, testset):
    if iterations >= self.sma_start_iter and iterations <= self.sma_end_iter:
        self.average_count += 1
        for param_q, param_k in zip(self.parameters(), self.model_sma.parameters()):
            param_k.data = (param_k.data * self.average_count + param_q.data) / (1.0 + self.average_count)
    else:
        for param_q, param_k in zip(self.parameters(), self.model_sma.parameters()):
            param_k.data = param_q.data


def _load_weight(self, best_params):
    """Copy the parameter weights of the moving average model before evaluating the test set"""
    for param_q, param_k in zip(self.parameters(), self.model_sma.parameters()):
        param_q.data = param_k.data


@MODELS.register_module('maensemble')
class MAEnsemble(BaseModel):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        sma_start_iter: Optional[int] = 0,
        sma_end_iter: Optional[int] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """The implementation of moving average model described in https://arxiv.org/pdf/2110.10832v2.

        Args:
            model (BaseModel): the model for which a moving average of the model parameters is maintained.
            sma_start_iter (int, optional): the choice of iteration (not epoch) to start model averaging.
            sma_end_iter (int, optional): the choice of iteration (not epoch) to stop model averaging.
            output_dir (Path)
            checkpoint_dir (Path)
        """

        nn.Module.__init__(self)
        self.model = model
        self.estimators = None
        self.model.sma_start_iter = sma_start_iter
        self.model.sma_end_iter = sma_end_iter if sma_end_iter is not None else float("inf")
        assert self.model.sma_start_iter <= self.model.sma_end_iter, 'sma_end_iter should be larger than sma_start_iter'
        self.model.average_count = 0
        self.model.estimators = None
        self.batch_size = self.model.batch_size
        self.out_ranges = self.model.out_ranges
        self.metrics = self.model.metrics
        self.metric_fn = self.model.metric_fn
        self.loss_fn = self.model.loss_fn
        self.checkpoint_dir = checkpoint_dir
        self.model.checkpoint_dir = checkpoint_dir
        self.model.output_dir = output_dir
        self.model._init_scheduler = types.MethodType(_init_scheduler, self.model)
        self.model._post_batch = types.MethodType(_post_batch, self.model)
        self.model._load_weight = types.MethodType(_load_weight, self.model)
        self.model.__deepcopy__ = types.MethodType(__deepcopy__, self.model)
        self.model.model_sma = self.model.__deepcopy__()  # the moving average network

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:

        self.model.fit(trainset, validset, testset)

        with torch.no_grad():
            metric_res = self.model.evaluate(trainset)
            best_res = {"train": metric_res}

        if validset is not None:
            print("Begin evaluate ensemble on vaildset ...")
            with torch.no_grad():
                eval_res = self.model.evaluate(validset)
            best_res["valid"] = eval_res

        if testset is not None:
            testset.load()
            print("Begin evaluate ensemble on testset ...")
            with torch.no_grad():
                test_res = self.model.evaluate(testset)
            best_res["test"] = test_res
            testset.freeup()

        with open(f"{self.checkpoint_dir}/ensemble_res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)

        torch.save(copy.deepcopy(self.state_dict()), f"{self.checkpoint_dir}/ensemble_net.pkl")

    def forward(self, inputs):

        self.model.forward(inputs)

    def predict(self, dataset: Dataset, name: str):

        self.model.predict(dataset, name)
