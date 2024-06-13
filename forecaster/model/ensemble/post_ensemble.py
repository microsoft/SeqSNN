from typing import Optional, List
from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ...common.utils import __deepcopy__

from ..base import MODELS, BaseModel


@MODELS.register_module("moe")
class MOEDispatcher(nn.Module):
    def __init__(
        self,
        n_estimators: Optional[int] = 3,
        hidden_size: Optional[int] = 64,
        feature_encoder: Optional[nn.Module] = None,
    ):
        """
        The dispatcher used by the Mixture-of-Expert.

        Args:
            n_estimators (int): the number of base models within the ensemble.
            hidden_size (int): the size of the embedding output from the feature extractor.
            feature_encoder (nn.Module): to embed inputs.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.feature_encoder = feature_encoder
        self.gate = torch.nn.Linear(hidden_size, n_estimators)

    def forward(self, inputs):
        _, input_embedding = self.feature_encoder(inputs)
        ensemble_weights = self.gate(input_embedding)
        return torch.nn.functional.softmax(ensemble_weights, dim=-1)


@MODELS.register_module("post_ensemble")
class PostEnsemble(BaseModel):
    def __init__(
        self,
        loss_fn: str,
        metrics: List[str],
        observe: str,
        lr: float = 1e-3,
        lower_is_better: bool = True,
        max_epoches: int = 50,
        batch_size: int = 512,
        early_stop: int = 10,
        optimizer: str = "Adam",
        weight_decay: float = 1e-5,
        n_estimators: Optional[int] = 3,
        model_pool_dir: Optional[Path] = None,
        ensemble_type: Optional[str] = 'average_ensemble',
        dispatcher: Optional[nn.Module] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Ensemble of trained models.
        Most of inputs are taken from BaseModel.

        Args:
            ensemble_type (str)
            dispatcher: network that generates ensemble weights
            loss_fn: to support flexible loss functions for training the ensemble network
        """

        nn.Module.__init__(self)
        if not hasattr(self, "hyper_paras"):
            self.hyper_paras = {}
        self.model_pool_dir = model_pool_dir
        self.ensemble_size = n_estimators
        self.ensemble_type = ensemble_type
        self.loss_fn = loss_fn
        self.network = dispatcher
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.out_ranges = None

        self._init_optimization(
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            metrics=metrics,
            observe=observe,
            lower_is_better=lower_is_better,
            max_epoches=max_epoches,
            batch_size=batch_size,
            early_stop=early_stop,
        )

        self._init_logger(output_dir)
        if torch.cuda.is_available():
            print("Using GPU")
            self.cuda()

    def forward(self, data):
        """
        Weight model predictions as ensemble output
        """
        features, preds = data
        class_type = int(preds.size(1) / self.ensemble_size)

        if self.ensemble_type == 'average_ensemble':
            ensemble_pred = torch.cat([torch.sum(preds[:, i::class_type], dim=1).unsqueeze(1) for i in range(class_type)], dim=1)
            ensemble_pred = torch.softmax(ensemble_pred, dim=-1)
        else:
            ensemble_weights = self.network(features)
            ensemble_pred = [torch.sum(preds[:, i::class_type] * ensemble_weights, dim=1).unsqueeze(1) for i in range(class_type)]
            ensemble_pred = torch.cat(ensemble_pred, dim=1)

        return ensemble_pred.float()

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:
        """Fit the ensemble network to data, if the ensemble type is not average_ensemble
           else run BaseModel.fit()

        Args:
            trainset (Dataset): The training dataset.
            validset (Dataset, optional): The evaluation dataset. Defaults to None.
            testset (Dataset, optional): The test dataset. Defaults to None.

        Returns:
            nn.Module: return the model itself.
        """

        if self.ensemble_type == 'average_ensemble':
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
        else:
            return super().fit(trainset, validset, testset)
