import json
import pickle as pkl
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket

from .base import MODELS


@MODELS.register_module()
class RocketPredictor:
    """
    The ROCKET time-series forecasting model described in https://arxiv.org/abs/1910.13051.
    """

    def __init__(
        self,
        task: str,
        weight_decay: float,
        num_kernels: int = 10000,
        model_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        if task == "regression":
            self.model = make_pipeline(Rocket(num_kernels=num_kernels), Ridge(alpha=weight_decay, normalize=True))
            self.care = "loss"
        else:
            self.model = make_pipeline(
                Rocket(num_kernels=num_kernels), RidgeClassifier(alpha=weight_decay, normalize=True)
            )
            self.care = "accuracy"

        self.checkpoint_dir = checkpoint_dir
        if model_path:
            with open(model_path, "rb") as f:
                self.model.set_params(**pkl.load(f))

    def fit(self, trainset, testset):
        X, y = trainset.get_values()
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        X, y = testset.get_values()
        test_score = self.model.score(X, y)
        res = {"train": {self.care: train_score}, "test": {self.care: test_score}}
        with open(f"{self.checkpoint_dir}/model.pkl", "wb") as f:
            pkl.dump(self.model.get_params(), f)

        with open(f"{self.checkpoint_dir}/res.json", "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)

    def predict(self, dataset, name):
        X, _ = dataset.get_values()
        prediction = self.model.predict(X)

        data_length = len(dataset.get_index())
        prediction = prediction.reshape(data_length, -1)

        prediction = pd.DataFrame(data=prediction, index=dataset.get_index())
        prediction.to_pickle(self.checkpoint_dir / (name + "_pre.pkl"))
        return prediction
