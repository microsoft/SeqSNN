from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

from ...module.basic_transform import Chomp2d
from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ..base import NETWORKS


class SpikeTemporalBlock2D(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        num_steps=4,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=(1, dilation),
            )
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.chomp1 = Chomp2d(padding)
        self.lif1 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=(1, dilation),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.chomp2 = Chomp2d(padding)
        self.lif2 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out1 = self.chomp1(self.bn1(self.conv1(x)))
        spk_rec1 = []
        for _ in range(self.num_steps):
            spk = self.lif1(out1)
            spk_rec1.append(spk)
        spks1 = torch.stack(spk_rec1, dim=-1)  # spks1: B, H, C, L, T
        spks1 = spks1.mean(-1)  # spks1: B, H, C, L

        out2 = self.chomp2(self.bn2(self.conv2(spks1)))
        spk_rec2 = []
        for _ in range(self.num_steps):
            spk = self.lif2(out2)
            spk_rec2.append(spk)
        spks2 = torch.stack(spk_rec2, dim=-1)  # spks2: B, H, C, L, T
        spks2 = spks2.mean(-1)  # spks2: B, H, C, L

        if torch.isnan(spks2).any() or torch.isinf(spks2).any():
            print("illegal value in TemporalBlock2D")

        if self.downsample is None:
            res = x
        else:
            res = self.downsample(x)
        spk_rec3 = []
        for _ in range(self.num_steps):
            spk = self.lif(spks2 + res)
            spk_rec3.append(spk)

        res = torch.stack(spk_rec3, dim=-1)  # res: B, H, C, L, T
        res = res.mean(-1)

        return res


@NETWORKS.register_module("SNN_TCN2D")
class SpikeTemporalConvNet2D(nn.Module):
    _snn_backend = "snntorch"

    def __init__(
        self,
        num_levels: int,
        channel: int,
        dilation: int,
        stride: int = 1,
        num_steps: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or "concat"
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
    ):
        """
        Args:
            num_channels: The number of convolutional channels in each layer.
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)

        self.num_steps = num_steps
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
        layers = []
        num_channels = [channel] * num_levels
        num_channels.append(1)
        for i in range(num_levels + 1):
            dilation_size = dilation**i
            in_channels = hidden_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                SpikeTemporalBlock2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    num_steps=num_steps,
                )
            ]

        self.network = nn.Sequential(*layers)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.__output_size = input_size + num_pe_neuron
        else:
            self.__output_size = input_size

    def forward(self, inputs: torch.Tensor):
        utils.reset(self.encoder)
        for layer in self.network:
            utils.reset(layer)

        inputs = self.encoder(inputs)  # B, H, C, L
        if self.pe_type != "none":
            # B, H, C, L -> H B L C' -> B H C' L
            inputs = self.pe(inputs.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        spks = self.network(inputs)
        spks = spks.squeeze(1)  # B, C', L
        return spks, spks[:, :, -1]  # [B, C', L], [B, C']

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
