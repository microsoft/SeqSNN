from typing import List, Optional
from pathlib import Path
import datetime
import copy
import time
import json

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from utilsd import use_cuda
from utilsd.earlystop import EarlyStop, EarlyStopStatus

from .base import MODELS, BaseModel
from ..common.utils import AverageMeter, to_torch
from ..common.function import printt


@MODELS.register_module()
class CPC(BaseModel):
    def __init__(
        self,
        optimizer: str,
        lr: float,
        weight_decay: float,
        max_epoches: int,
        batch_size: int,
        forecast_step: int = 10,
        network: Optional[nn.Module] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        early_stop: Optional[int] = None,
        model_path: Optional[str] = None,
    ):
        """
        The implementation of representation learning with contrastive predictive coding from http://arxiv.org/abs/1807.03748.
        forecast_step: the number of steps for cpc to forecast
        """
        self.hyper_paras = {
            "forecast_step": forecast_step,
        }
        loss_fn = metrics = observe = None
        lower_is_better = True
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

    def _init_optimization(
        self,
        optimizer: str,
        lr: float,
        weight_decay: float,
        lower_is_better: bool,
        max_epoches: int,
        batch_size: int,
        early_stop: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Setup loss function, evaluation metrics and optimizer"""
        for k, v in locals().items():
            if k not in ["self", "metrics", "observe", "lower_is_better", "loss_fn"]:
                self.hyper_paras[k] = v
        if early_stop is not None:
            self.early_stop = EarlyStop(patience=early_stop, mode="min" if lower_is_better else "max")
        else:
            self.early_stop = EarlyStop(patience=max_epoches)
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_network(self, network, forecast_step: int) -> None:
        self.network = network
        self.forecast_step = forecast_step
        hidden_size = network.hidden_size
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(forecast_step)])

    def forward(self, inputs):
        z, g = self.network.get_cpc_repre(inputs)
        forecasts = []
        for i in range(self.forecast_step):
            forecasts.append(self.W[i](g))
        return z, forecasts

    def _loss_k(self, z, forecast, k) -> torch.Tensor:
        forecast = forecast[:, :-k, :]  # B, L-k, H
        z = z[:, k:, :]  # B, L-k, H
        # res: B, B, L-k
        res = torch.einsum("ijk,ljk->ilj", (forecast, z))  # B, B, L-k
        res = torch.diagonal(self.lsoftmax(res), dim1=0, dim2=1)  # L-k, B
        assert not torch.isnan(res).any() and not torch.isinf(res).any(), "{res}"
        return -res.mean()

    def get_loss_k(self, z, forecast, k) -> torch.Tensor:
        forecast = forecast[:, :-k, :]  # B, L-k, H
        z = z[:, k:, :]  # B, L-k, H
        # res: B, B, L-k
        res = torch.einsum("ijk,ljk->ilj", (forecast, z))  # B, B, L-k
        res = torch.diagonal(self.lsoftmax(res), dim1=0, dim2=1)  # L-k, B
        return res

    def _loss(self, z, forecasts) -> torch.Tensor:
        assert self.forecast_step > 0
        return sum([self._loss_k(z, forecasts[i], i + 1) for i in range(self.forecast_step)])

    def fit(self, trainset, validset=None, testset=None) -> nn.Module:
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

        epoch = 0
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        start_epoch, best_res = self._resume()

        for epoch in range(start_epoch, self.max_epoches):
            # if self.early_stop_epoches is not None and stop_epoches >= self.early_stop_epoches:
            #     print("earlystop")
            #     break
            # training
            self.train()
            train_loss = AverageMeter()
            start_time = time.time()
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data = to_torch(data, device="cuda")
                z, forecasts = self(data)
                loss = self._loss(z, forecasts)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss = loss.item()
                train_loss.update(loss, np.prod(label.shape))

            train_time = time.time() - start_time
            loss = train_loss.performance()  # loss
            start_time = time.time()
            metric_res = {"loss": loss}

            # print log
            printt(f"{epoch}\t'train'\tTime:{train_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/pretrain", v, epoch)
            self.writer.flush()

            if validset is not None:
                with torch.no_grad():
                    eval_res = self.evaluate(validset, epoch)
                value = eval_res["loss"]
                es = self.early_stop.step(value)
                if es == EarlyStopStatus.BEST:
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res, "valid": eval_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/cpc_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/cpc_network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            else:
                es = self.early_stop.step(metric_res["loss"])
                if es == EarlyStopStatus.BEST:
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/cpc_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/cpc_network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            self._checkpoint(epoch, best_res)

        # finish training, test, save model and write logs
        self.load_state_dict(self.best_params, strict=True)
        trainset.freeup()
        if validset is not None:
            validset.freeup()
        if testset is not None:
            testset.load()
            with torch.no_grad():
                test_res = self.evaluate(testset)
            for k, v in test_res.items():
                self.writer.add_scalar(f"{k}/pretrain_test", v, epoch)
            value = test_res["loss"]
            best_res["test"] = test_res
            testset.freeup()
        torch.save(self.best_params, f"{self.checkpoint_dir}/cpc_best.pkl")
        torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
        with open(f"{self.checkpoint_dir}/pretrain_res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)

        return self

    def _checkpoint(self, cur_epoch, best_res):
        torch.save(
            {
                "earlystop": self.early_stop.state_dict(),
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "epoch": cur_epoch,
                "best_res": best_res,
                "best_params": self.best_params,
                "best_network_params": self.best_network_params,
            },
            self.checkpoint_dir / "cpc_resume.pth",
        )
        print(f"Checkpoint saved to {self.checkpoint_dir / 'cpc_resume.pth'}", __name__)

    def _resume(self):
        if (self.checkpoint_dir / "cpc_resume.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'cpc_resume.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "cpc_resume.pth")
            self.early_stop.load_state_dict(checkpoint["earlystop"])
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            self.best_params = checkpoint["best_params"]
            self.best_network_params = checkpoint["best_network_params"]
            return checkpoint["epoch"], checkpoint["best_res"]
        else:
            print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
            return 0, {}

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:
        loader = DataLoader(
            validset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.eval()
        eval_loss = AverageMeter()
        start_time = time.time()
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data = to_torch(data, device="cuda")
                z, forecasts = self(data)
                loss = self._loss(z, forecasts)
                loss = loss.item()
                eval_loss.update(loss, np.prod(label.shape))

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        start_time = time.time()
        metric_res = {}
        metric_res["loss"] = loss

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/pretrain_valid", v, epoch)

        return metric_res
