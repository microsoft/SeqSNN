from typing import Optional
from pathlib import Path

import snntorch as snn
from snntorch import surrogate
from snntorch import utils

import torch
from torch import nn

from ..base import NETWORKS

from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder


class SNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        beta: float = 0.99,
        output_mems: bool = False,
    ):
        super().__init__()
        self.spike_grad = surrogate.atan(alpha=2.0)
        self.input_size = input_size
        self.num_steps = num_steps
        self.beta = beta
        self.full_rec = output_mems
        self.lif = snn.Leaky(
            beta=self.beta,
            spike_grad=self.spike_grad,
            init_hidden=True,
            output=output_mems,
        )
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        if inputs.size(-1) == self.input_size:
            # assume static spikes:
            cur = self.linear(inputs)
            static = True
        elif inputs.size(-1) == self.num_steps and inputs.size(-2) == self.input_size:
            # assume dynamic spikes:
            cur = self.linear(inputs.transpose(-1, -2)).transpose(-1, -2)  # BC, H, T
            static = False
        else:
            raise ValueError(
                f"Input size mismatch! "
                f"Got {inputs.size()} but expected (..., {self.input_size}, {self.num_steps}) or (..., {self.input_size})"
            )
        spk_rec = []
        mem_rec = []
        if self.full_rec:
            for i_step in range(self.num_steps):
                if static:
                    spk, mem = self.lif(cur)
                else:
                    spk, mem = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
                mem_rec.append(mem)
            spks = torch.stack(spk_rec, dim=-1)
            mems = torch.stack(mem_rec, dim=-1)
            return spks, mems
        else:
            for i_step in range(self.num_steps):
                if static:
                    spk = self.lif(cur)
                else:
                    spk = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
            spks = torch.stack(spk_rec, dim=-1)
            return spks


@NETWORKS.register_module("SNN")
class TSSNN(nn.Module):
    _snn_backend = "snntorch"

    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)

        self.net = nn.Sequential(
            *[
                SNNCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        for layer in self.net:
            utils.reset(layer)
        hiddens = self.encoder(inputs)  # B, L, H
        _, t, _ = hiddens.size()  # B, L, H
        for i in range(t):
            spks, _ = self.net(hiddens[:, i, :])
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * H, B * H

    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("SNN2d")
class TSSNN2D(nn.Module):
    _snn_backend = "snntorch"

    def __init__(
        self,
        hidden_size: int,
        encoder_dim: int = 8,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
    ):
        super().__init__()
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.hidden_size = hidden_size
        self.temporal_encoder = SpikeEncoder[self._snn_backend][encoder_type](
            encoder_dim
        )

        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=num_steps,
        )
        self.encoder = nn.Linear(encoder_dim, self.hidden_size)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.__output_size = hidden_size * (input_size + num_pe_neuron)
        else:
            self.__output_size = hidden_size * num_steps

        self.net = nn.Sequential(
            *[
                SNNCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        utils.reset(self.temporal_encoder)
        for layer in self.net:
            utils.reset(layer)
        h = self.temporal_encoder(inputs)  # B, H_encoder, C, L
        h = self.encoder(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.pe_type != "none":
            h = self.pe(h.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        bs, hidden_size, c_num, length = h.size()
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * CH, B * CH

    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("ISNN2d")
class ITSSNN2D(nn.Module):
    _snn_backend = "snntorch"

    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)
        self.net = nn.Sequential(
            *[
                SNNCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        # self.__output_size = hidden_size * input_size
        self.__output_size = hidden_size * num_steps

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # C: number of variate (tokens), can also includes covariates

        utils.reset(self.encoder)
        for layer in self.net:
            utils.reset(layer)
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        spks = spks.reshape(bs, c_num, -1)  # B, C, H*Time Step
        return spks, spks  # [B, C, H*Time Step], [B, C, H*Time Step]

    @property
    def output_size(self):
        return self.__output_size
