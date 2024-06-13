from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np

import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils


from .base import MODELS, BaseModel


@MODELS.register_module("snnts", inherit=True)
class SNN_TS(BaseModel):
    def __init__(
        self,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
        out_size: Optional[int] = None,
        aggregate: bool = True,
        **kwargs,
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
            **kwargs
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
        if task == "classification":
            self.act_out = nn.Sigmoid()
            out_size = 1
        elif task == "multiclassification":
            self.act_out = nn.LogSoftmax(-1)
        elif task == "regression":
            self.act_out = nn.Identity()
        else:
            raise ValueError(
                ("Task must be 'classification', 'multiclassification', 'regression'")
            )

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

        if out_size is not None:
            self.fc_out = nn.Linear(network.output_size, out_size)
        else:
            self.fc_out = nn.Identity()
        # self.fc_lif = snn.Leaky(beta=0.99, spike_grad=surrogate.fast_sigmoid(slope=25.), init_hidden=True, output=True)
        # self.last = nn.Linear(out_size, out_size)
        
    def forward(self, inputs):
        # seq_out, emb_outs = self.network(inputs)  # [B, T, H], [B, out_size])
        # print(seq_out.size(), emb_outs.size())
        # spks, mems = [], []
        # for _ in range(seq_out.size(1)):
        #     spk, mem = self.fc_lif(self.fc_out(seq_out[:, _, :]))
        #     spks.append(spk)
        #     mems.append(mem)
        # preds = self.act_out(self.last(mems[-1]).squeeze(-1))
        # return preds

        # print(inputs.shape) # B, L, C
        seq_out, emb_outs = self.network(inputs)  # SNN: [B, Time Step, H], [B, H]
        
        if self.aggregate:
            out = emb_outs # SNN, [B, H]
            preds = self.act_out(self.fc_out(out).squeeze(-1))
            return preds # [B, Out_size]
        else:
            out = seq_out # SNN, [B, T, H]
            preds = self.act_out(self.fc_out(out.mean(1).squeeze(-1)).squeeze(-1)) # T average
            return preds # [B, Out_size]
