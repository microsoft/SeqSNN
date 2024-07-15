from typing import Optional

import torch

from .base import RUNNERS
from .timeseries import TS


@RUNNERS.register_module("vsts", inherit=True)
class VariateSpecTS(TS):
    def __init__(
        self,
        denormalize: bool = False,
        valid_variates: Optional[int] = None,
        **kwargs,
    ):
        """
        The model for variate specific time-series prediction.
        This means each variate has its own embedding and temporal encoder.

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
            denormalize: whether to denormalize the output.
            valid_variates: the number of valid variates to output. If None, output all variates.
        """
        super().__init__(**kwargs)
        self.denormalize = denormalize
        self.valid_variates = valid_variates

    def forward(self, inputs: torch.Tensor):
        B, _, C = inputs.size()
        # introduced by itransformer
        if self.denormalize:
            means = inputs.mean(1, keepdim=True).detach()
            inputs = inputs - means
            std = torch.sqrt(
                torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            inputs /= std
        seq_out, emb_outs = self.network(
            inputs
        )  # [B, N, E], [B, N, E], C=N: number of variate, E:hidden_size
        if self.aggregate:
            out = emb_outs
        else:
            out = seq_out
        out = out.reshape(B, C, -1)  # [B, N, E], C=N
        preds = self.act_out(self.fc_out(out).squeeze(-1)).permute(0, 2, 1)  # [B, O, N]
        if self.denormalize:
            preds = preds * (
                std[:, 0, :].unsqueeze(1).repeat(1, self.hyper_paras["out_size"], 1)
            )
            preds = preds + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.hyper_paras["out_size"], 1)
            )
        if self.valid_variates is not None:
            preds = preds[:, :, : self.valid_variates]
        preds = preds[:, -self.hyper_paras["out_size"] :, :]
        return preds.reshape(B, -1)  # [B, O*N], O*N = horizon * variate num
