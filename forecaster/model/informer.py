from typing import Tuple, Union, List, Optional
from .base import MODELS, BaseModel
from pathlib import Path
import datetime
import copy
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from utilsd import use_cuda
from utilsd.earlystop import EarlyStop, EarlyStopStatus

from forecaster.common.function import get_loss_fn, get_metric_fn, printt
from forecaster.common.utils import AverageMeter, GlobalTracker, to_torch


@MODELS.register_module("Informer", inherit=True)
class Informer(BaseModel):
    def __init__(
        self,
        task: str,
        loss_fn: str,
        metrics: List[str],
        optimizer: str,
        observe: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lower_is_better: bool = True,
        max_epoches: int = 50,
        batch_size: int = 512,
        network: Optional[nn.Module] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        early_stop: int = 30,
        out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
        model_path: Optional[str] = None,
        aggregate: bool = True,
        out_size: Optional[int] = None,
    ):
        """
        The model for general time-series prediction.

        Args:
            task: the prediction task, classification or regression.
            optimizer: which optimizer to use.
            lr: learning rate.
            weight_decay: L2 normlize weight
            loss_fn: loss function.
            metrics: metrics to evaluate model.
            observe: metric for model selection (earlystop).
            lower_is_better: whether a lower observed metric means better result.
            max_epoches: maximum epoch to learn.
            batch_size: batch size.
            early_stop: earlystop rounds.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            model_path: the path to existing model parameters for continued training or finetuning
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """
        self.hyper_paras = {
            "task": task,
            "out_ranges": out_ranges,
            "out_size": out_size,
            "aggregate": aggregate,
        }
        super().__init__(
            loss_fn,
            metrics,
            observe,
            lr,
            lower_is_better,
            max_epoches,
            batch_size,
            early_stop,
            optimizer,
            weight_decay,
            network,
            model_path,
            output_dir,
            checkpoint_dir,
        )

    def _build_network(
        self,
        network,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int, int], Tuple[int, int]]]] = None,
        out_size: Optional[int] = None,
        aggregate: bool = True,
    ) -> None:
        """Initilize the network parameters

        Args:
            task: the prediction task, classification or regression.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """

        self.network = network
        self.aggregate = aggregate

        # Output

        if out_ranges is not None:
            self.out_ranges = []
            for ran in out_ranges:
                if len(ran) == 2:
                    self.out_ranges.append(np.arange(ran[0], ran[1]))
                elif len(ran) == 3:
                    self.out_ranges.append(np.arange(ran[0], ran[1], ran[2]))
                else:
                    raise ValueError(f"Unknown range {ran}")
            self.out_ranges = np.concatenate(self.out_ranges)
        else:
            self.out_ranges = None

    def forward(self, *inputs):
        out = self.network(*inputs)  # [B, T, H]
        preds = out.squeeze(-1)
        return preds

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:
        """Fit the model to data, if evaluation dataset is offered,
           model selection (early stopping) would be conducted on it.

        Args:
            trainset (Dataset): The training dataset.
            validset (Dataset, optional): The evaluation dataset. Defaults to None.
            testset (Dataset, optional): The test dataset. Defaults to None.

        Returns:
            nn.Module: return the model itself.
        """
        trainset.load()
        if validset is not None:
            validset.load()

        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

        self._init_scheduler(len(loader))
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        iterations = 0
        start_epoch, best_res = self._resume()
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best
        for epoch in range(start_epoch, self.max_epoches):
            self.train()
            train_loss = AverageMeter()
            train_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
            start_time = time.time()
            for _, (x, label, x_mark, y_mark) in enumerate(loader):
                if use_cuda():
                    x, label, x_mark, y_mark = (
                        to_torch(x, device="cuda"),
                        to_torch(label, device="cuda"),
                        to_torch(x_mark, device="cuda"),
                        to_torch(y_mark, device="cuda"),
                    )
                # print(x.shape)
                # print(label.shape)
                # print(x_mark.shape)
                # print(y_mark.shape)
                pred = self(x, x_mark, torch.zeros_like(label), y_mark)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss = loss.item()
                train_loss.update(loss, np.prod(label.shape))
                train_global_tracker.update(label, pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                iterations += 1
                self._post_batch(iterations, epoch, train_loss, train_global_tracker, validset, testset)

            train_time = time.time() - start_time
            loss = train_loss.performance()  # loss
            start_time = time.time()
            train_global_tracker.concat()
            metric_res = train_global_tracker.performance()
            metric_time = time.time() - start_time
            metric_res["loss"] = loss

            # print log
            printt(f"{epoch}\t'train'\tTime:{train_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/train", v, epoch)
            self.writer.flush()

            if validset is not None:
                with torch.no_grad():
                    eval_res = self.evaluate(validset, epoch)
                value = eval_res[self.observe]
                es = self.early_stop.step(value)
                if es == EarlyStopStatus.BEST:
                    best_score = value
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res, "valid": eval_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            else:
                es = self.early_stop.step(metric_res[self.observe])
                if es == EarlyStopStatus.BEST:
                    best_score = metric_res[self.observe]
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            self._checkpoint(epoch, {**best_res, "best_epoch": best_epoch})

        # release the space of train and valid dataset
        trainset.freeup()
        if validset is not None:
            validset.freeup()

        # finish training, test, save model and write logs
        self._load_weight(self.best_params)
        if testset is not None:
            testset.load()
            print("Begin evaluate on testset ...")
            with torch.no_grad():
                test_res = self.evaluate(testset)
            for k, v in test_res.items():
                self.writer.add_scalar(f"{k}/test", v, epoch)
            value = test_res[self.observe]
            best_score = value
            best_res["test"] = test_res
            testset.freeup()
        torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
        torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
        with open(f"{self.checkpoint_dir}/res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)
        keys = list(self.hyper_paras.keys())
        for k in keys:
            if type(self.hyper_paras[k]) not in [int, float, str, bool, torch.Tensor]:
                self.hyper_paras.pop(k)
        self.writer.add_hparams(self.hyper_paras, {"result": best_score, "best_epoch": best_epoch})

        return self

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:
        """Evaluate the model on the given dataset.

        Args:
            validset (Dataset): The dataset to be evaluated on.
            epoch (int, optional): If given, would write log to tensorboard and stdout. Defaults to None.

        Returns:
            dict: The results of evaluation.
        """
        loader = DataLoader(
            validset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.eval()
        eval_loss = AverageMeter()
        eval_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
        start_time = time.time()
        validset.load()
        with torch.no_grad():
            for _, (x, label, x_mark, y_mark) in enumerate(loader):
                if use_cuda():
                    x, label, x_mark, y_mark = (
                        to_torch(x, device="cuda"),
                        to_torch(label, device="cuda"),
                        to_torch(x_mark, device="cuda"),
                        to_torch(y_mark, device="cuda"),
                    )
                pred = self(x, x_mark, torch.zeros_like(label), y_mark)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                loss = loss.item()
                eval_loss.update(loss, np.prod(label.shape))
                eval_global_tracker.update(label, pred)

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        start_time = time.time()
        eval_global_tracker.concat()
        metric_res = eval_global_tracker.performance()
        metric_time = time.time() - start_time
        metric_res["loss"] = loss

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/valid", v, epoch)

        return metric_res

    def predict(self, dataset: Dataset, name: str):
        """Output the prediction on given data.

        Args:
            datasets (Dataset): The dataset to predict on.
            name (str): The results would be saved to {name}_pre.pkl.

        Returns:
            np.ndarray: The model output.
        """
        self.eval()
        preds = []
        dataset.load()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for _, (x, label, x_mark, y_mark) in enumerate(loader):
            if use_cuda():
                x, label, x_mark, y_mark = (
                    to_torch(x, device="cuda"),
                    to_torch(label, device="cuda"),
                    to_torch(x_mark, device="cuda"),
                    to_torch(y_mark, device="cuda"),
                )
            pred = self(x, x_mark, torch.zeros_like(label), y_mark)
            if self.out_ranges is not None:
                pred = pred[:, self.out_ranges]
            pred = pred.squeeze().cpu().detach().numpy()
            preds.append(pred)

        prediction = np.concatenate(preds, axis=0)
        data_length = len(dataset.get_index())
        prediction = prediction.reshape(data_length, -1)

        prediction = pd.DataFrame(data=prediction, index=dataset.get_index())
        prediction.to_pickle(self.checkpoint_dir / (name + "_pre.pkl"))
        return prediction