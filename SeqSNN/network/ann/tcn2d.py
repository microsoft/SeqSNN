from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm


from SeqSNN.module import Chomp2d, PositionEmbedding
from ..base import NETWORKS


class TemporalBlock2D(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()
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
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

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
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        )
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("illegal value in TemporalBlock2D")
        res = x if self.downsample is None else self.downsample(x)
        res = self.relu(out + res)
        return res


@NETWORKS.register_module("TCN2D")
class TemporalConvNet2D(nn.Module):
    def __init__(
        self,
        num_levels: int,
        channel: int,
        dilation: int,
        position_embedding: bool,
        emb_type: str,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """The implementation of TCN described in https://arxiv.org/abs/1803.01271.

        Args:
            num_channels: The number of convolutional channels in each layer.
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding to use. Can be "learn" or "static".
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        super().__init__(self)
        layers = []
        num_channels = [channel] * num_levels
        num_channels.append(1)
        for i in range(num_levels + 1):
            dilation_size = dilation**i
            in_channels = 1 if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.__output_size = input_size
        self._position_embedding = position_embedding

        if position_embedding:
            self.emb = PositionEmbedding(
                emb_type, input_size, max_length, dropout=dropout
            )

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs: torch.Tensor):
        if self._position_embedding:
            inputs = self.emb(inputs)
        inputs = inputs.unsqueeze(1).transpose(2, 3)
        hiddens = self.network(inputs)  # B, 1, C, L
        return hiddens, hiddens[:, :, :, -1].squeeze(1)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
