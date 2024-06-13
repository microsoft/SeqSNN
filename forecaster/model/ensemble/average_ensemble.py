from typing import Optional, List
from pathlib import Path
import types
import copy
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ...common.utils import __deepcopy__

from ..base import MODELS, BaseModel


def _init_weight(m):
    """Reinitialize the network parameters to train the model with different initialization."""
    classname = m.__class__.__name__
    if (
        classname.find("Conv") != -1
        and classname.find("Layer") == -1
        and classname.find("Decoder") == -1
        and classname.find("Encoder") == -1
    ):
        nn.init.xavier_uniform_(m.weight)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


@MODELS.register_module("average_ensemble")
class AverageEnsemble(BaseModel):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_estimators: Optional[int] = 3,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        AverageEnsemble averages predictions of multiple single models trained with different random initializations.

        Reference:
            B. Lakshminarayanan, A. Pritzel, C. Blundell., Simple and Scalable
            Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017.

        Args:
            model (BaseModel): the model for snapshot ensemble.
            n_estimators (int, optional): the number of base models.
            output_dir (Path)
            checkpoint_dir (Path)
        """

        nn.Module.__init__(self)
        self.n_estimators = n_estimators  # the number of base models
        self.estimators = nn.ModuleList()  # the list of trained base models for ensembling
        model.__deepcopy__ = types.MethodType(__deepcopy__, model)
        for _ in range(self.n_estimators):
            base_model = copy.deepcopy(model)
            base_model.load_state_dict(model.state_dict())
            self.estimators.append(base_model)
            model.apply(_init_weight)

        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = model.batch_size
        self.out_ranges = model.out_ranges
        self.metrics = model.metrics
        self.metric_fn = model.metric_fn
        self.loss_fn = model.loss_fn

    def forward(self, inputs):
        """If the models to be aggregated are ready, we average their predictions,
        otherwise we use a single model for prediction.
        """
        results = []
        for estimator in self.estimators:
            results.append(estimator(inputs))

        return sum(results) / len(results)

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:

        for i in range(self.n_estimators):
            self.estimators[i].checkpoint_dir = Path(self.checkpoint_dir) / f"model_{i+1}"
            if not os.path.exists(self.estimators[i].checkpoint_dir):
                os.makedirs(self.estimators[i].checkpoint_dir)
            self.estimators[i]._init_logger(self.estimators[i].checkpoint_dir)
            self.estimators[i].fit(trainset, validset, testset)

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
