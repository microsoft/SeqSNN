
import math

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class CPG(nn.Module):
    num_neurons: int = 10
    w_max: float = 10000.0
    l_max: int = 5000

    def __post_init__(self):
        self._cpg = torch.zeros(self.l_max, self.num_neurons)
        position = torch.arange(0, self.l_max, dtype=torch.float).unsqueeze(
            1
        )  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, self.num_neurons, 2).float()
            * (-math.log(self.w_max) / self.num_neurons)
        )
        div_term_single = torch.exp(
            torch.arange(0, self.num_neurons - 1, 2).float()
            * (-math.log(self.w_max) / self.num_neurons)
        )
        self._cpg[:, 0::2] = torch.heaviside(
            torch.sin(position * div_term) - 0.8, torch.tensor([1.0])
        )
        self._cpg[:, 1::2] = torch.heaviside(
            torch.cos(position * div_term_single) - 0.8, torch.tensor([1.0])
        )

    @property
    def cpg(self):
        return self._cpg


class CPGLinear(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, cpg: CPG = CPG(), dropout: float = 0.1
    ):
        super().__init__()
        self.cpg = nn.Parameter(cpg.cpg, requires_grad=False)
        self.inp_linear = nn.Linear(input_size, output_size)
        self.cpg_linear = nn.Linear(cpg.num_neurons, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # B TL D
        cpg = self.cpg[: x.size(-2)]
        x = self.dropout(x)
        return self.inp_linear(x) + self.cpg_linear(cpg)
