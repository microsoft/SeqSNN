from typing import Optional
from pathlib import Path
import torch
from torch import nn

from spikingjelly.activation_based import surrogate, neuron, functional

from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ..base import NETWORKS


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class SpikeRNNCell(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        # T, B, L, C'
        T, B, L, _ = x.shape
        x = x.flatten(0, 1)  # TB, L, C'
        x = self.linear(x)
        x = x.reshape(T, B, L, -1)
        x = self.lif(x)  # T, B, L, C'
        return x


@NETWORKS.register_module("SpikeRNN")
class SpikeRNN(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 4,
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
        self.neuron_pe_scale = neuron_pe_scale
        self.temporal_encoder = SpikeEncoder[self._snn_backend][encoder_type](num_steps)

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

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.dim = hidden_size + num_pe_neuron
        else:
            self.dim = hidden_size

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.encoder = nn.Linear(input_size + num_pe_neuron, self.dim)
        else:
            self.encoder = nn.Linear(input_size, self.dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
            backend=backend,
        )

        self.net = nn.Sequential(
            *[
                SpikeRNNCell(input_size=self.dim, output_size=self.dim)
                for i in range(layers)
            ]
        )

        self.__output_size = self.dim

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        functional.reset_net(self)
        hiddens = self.temporal_encoder(inputs)  # T, B, C, L
        hiddens = hiddens.transpose(-2, -1)  # T, B, L, C
        T, B, L, _ = hiddens.size()  # T, B, L, D
        if self.pe_type != "none":
            hiddens = self.pe(hiddens)  # T B L C'
        hiddens = self.encoder(hiddens.flatten(0, 1)).reshape(T, B, L, -1)  # T B L D
        hiddens = self.init_lif(hiddens)
        hiddens = self.net(hiddens)  # T, B, L, D
        out = hiddens.mean(0)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.dim


@NETWORKS.register_module("SpikeRNN2d")
class SpikeRNN2D(nn.Module):
    _snn_backend = "spikingjelly"

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
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
    ):
        super().__init__()
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)

        self.net = nn.Sequential(
            *[
                SpikeRNNCell(
                    hidden_size,
                    hidden_size,
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size * input_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * CH, B * CH

    @property
    def output_size(self):
        return self.__output_size
