from typing import Optional, List
from pathlib import Path
import datetime
import copy
import math
import types
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ...common.function import printt
from ...common.utils import __deepcopy__

from ..base import MODELS, BaseModel


def _init_scheduler(self, loader_length):
    """
    Set the learning rate scheduler for snapshot ensemble.
    Please refer to the equation (2) in https://arxiv.org/abs/1704.00109 for details.
    Implementation of the function is referrfed to
    https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/torchensemble/snapshot_ensemble.py.
    """

    if self.max_epoches * loader_length < self.n_estimators:
        raise ValueError("The number of snapshots cannot be greater than the total number of iterations of the training data set.")
    self.n_iters_per_estimator = self.max_epoches * loader_length // self.n_estimators
    n_iters = self.max_epoches * loader_length
    T_M = math.ceil(n_iters / self.n_estimators)
    self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda iteration: 0.5 * (
        torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1
    ))


def _early_stop(self):
    return False


def _post_batch(self, iterations: int, epoch, train_loss, train_global_tracker, validset, testset):
    self.writer.add_scalar(f"{'lr'}/train", self.optimizer.state_dict()['param_groups'][0]['lr'], iterations)
    self.writer.flush()
    if iterations % self.n_iters_per_estimator == 0:
        # generate and save the snapshot
        snapshot = copy.deepcopy(self)
        snapshot.load_state_dict(self.state_dict())
        self.estimators.append(snapshot)

        loss = train_loss.performance()
        train_global_tracker.concat()
        metric_res = train_global_tracker.performance()
        metric_res["loss"] = loss

        # print log
        printt(f"trained ensemble {len(self.estimators)}\t'train'")
        for metric, value in metric_res.items():
            printt(f"{metric}: {value:.4f}")
        print(f"{datetime.datetime.today()}")

        if validset is not None:
            with torch.no_grad():
                eval_res = self.evaluate(validset, epoch)
            best_res = {"train": metric_res, "valid": eval_res}
        else:
            best_res = {"train": metric_res}

        if testset is not None:
            with torch.no_grad():
                test_res = self.evaluate(testset)
            best_res["test"] = test_res

        params = copy.deepcopy(self.state_dict())
        network_params = copy.deepcopy(self.network.state_dict())
        ensemble_checkpoint_dir = Path(self.checkpoint_dir) / f"model_{len(self.estimators)}"
        if not os.path.exists(ensemble_checkpoint_dir):
            os.makedirs(ensemble_checkpoint_dir)
        with open(f"{ensemble_checkpoint_dir}/res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)
        torch.save(params, f"{ensemble_checkpoint_dir}/model_best.pkl")
        torch.save(network_params, f"{ensemble_checkpoint_dir}/network_best.pkl")
        self._checkpoint(epoch, {**best_res, "best_epoch": epoch}, ensemble_checkpoint_dir)


def _load_weight(self, best_params):
    pass


@MODELS.register_module('snapshot_ensemble')
class SnapshotEnsemble(BaseModel):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_estimators: Optional[int] = 3,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Snapshot ensemble generates many base estimators by enforcing a base estimator to converge to its local minima many times and save the model parameters at that point as a snapshot. The final prediction takes the average over predictions from all snapshot models.
        Reference:
            G. Huang, Y.-X. Li, G. Pleiss et al., Snapshot Ensemble:
            Train 1, and M for free, ICLR, 2017.

        Args:
            model (BaseModel): the model for snapshot ensemble.
            n_estimators (int, optional): the number of snapshots.
            output_dir (Path)
            checkpoint_dir (Path)
        """

        nn.Module.__init__(self)
        self.model = model
        self.model.n_estimators = n_estimators  # the number of base models
        self.model.estimators = []  # the list of trained base models for ensembling
        self.batch_size = self.model.batch_size
        self.out_ranges = self.model.out_ranges
        self.metrics = model.metrics
        self.metric_fn = model.metric_fn
        self.loss_fn = model.loss_fn
        self.checkpoint_dir = checkpoint_dir
        self.model.checkpoint_dir = checkpoint_dir
        self.model.output_dir = output_dir
        self.model._early_stop = types.MethodType(_early_stop, self.model)
        self.model._init_scheduler = types.MethodType(_init_scheduler, self.model)
        self.model._post_batch = types.MethodType(_post_batch, self.model)
        self.model._load_weight = types.MethodType(_load_weight, self.model)
        self.model.__deepcopy__ = types.MethodType(__deepcopy__, self.model)
        self.model._init_logger(output_dir)

    def forward(self, inputs):
        """If the models to be aggregated are ready, we average their predictions,
        otherwise we use a single model for prediction.
        """
        results = []
        for estimator in self.model.estimators:
            results.append(estimator(inputs))

        return sum(results) / len(results)

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:

        self.model.fit(trainset, validset, testset)
        self.estimators = self.model.estimators

        with torch.no_grad():
            metric_res = self.evaluate(trainset)
            best_res = {"train": metric_res}

        if validset is not None:
            print("Begin evaluate ensemble on vaildset ...")
            with torch.no_grad():
                eval_res = self.evaluate(validset)
            best_res["valid"] = eval_res

        if testset is not None:
            testset.load()
            print("Begin evaluate ensemble on testset ...")
            with torch.no_grad():
                test_res = self.evaluate(testset)
            best_res["test"] = test_res
            testset.freeup()

        with open(f"{self.checkpoint_dir}/ensemble_res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)

        torch.save(copy.deepcopy(self.state_dict()), f"{self.checkpoint_dir}/ensemble_net.pkl")
