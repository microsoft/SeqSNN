from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_

from .tsrnn import NETWORKS, TSRNN
from ..module import PositionEmbedding, VanillaAttention


class ReluGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReluGRUCell, self).__init__()
        self.weight_ir = Parameter(xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        self.weight_hr = Parameter(xavier_uniform_(torch.Tensor(hidden_size, hidden_size)))
        self.bias_r = Parameter(torch.Tensor(hidden_size).fill_(0.01))
        self.weight_iz = Parameter(xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        self.weight_hz = Parameter(xavier_uniform_(torch.Tensor(hidden_size, hidden_size)))
        self.bias_z = Parameter(torch.Tensor(hidden_size).fill_(0.01))
        self.weight_in = Parameter(xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        self.bias_in = Parameter(torch.Tensor(hidden_size).fill_(0.01))
        self.weight_hn = Parameter(xavier_uniform_(torch.Tensor(hidden_size, hidden_size)))
        self.bias_hn = Parameter(torch.Tensor(hidden_size).fill_(0.01))

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(torch.matmul(input, self.weight_ir) + torch.matmul(hidden, self.weight_hr) + self.bias_r)
        z = torch.sigmoid(torch.matmul(input, self.weight_iz) + torch.matmul(hidden, self.weight_hz) + self.bias_z)
        n = F.relu(torch.matmul(input, self.weight_in) + torch.matmul(hidden * r, self.weight_hn) + self.bias_in)
        hidden = (1 - z) * n + z * hidden
        return hidden


@NETWORKS.register_module()
class LSTNetSkip(TSRNN):
    def __init__(
        self,
        position_embedding: bool,
        emb_type: str,
        hidden_size: int,
        channel_size: int,
        skip_size: int,
        autoregressive_size: int,
        kernel_size: int = 3,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """The LstNet-Skip network described in http://arxiv.org/abs/1703.07015.

        Args:
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding. Can be static or learn.
            hidden_size: Hidden dimension of the network
            channel_size: The channel size of convolution layer.
            skip_size: The number of hidden cells skipped through in recurrent-skip component.
            autoregressive_size: The input window of autoregressive component.
            kernel_size: Kernel size of convolution layer.
        """
        nn.Module.__init__(self)
        self.convolution = nn.Sequential(
            nn.Conv1d(input_size, channel_size, kernel_size, padding=(kernel_size - 1) // 2), nn.ReLU()
        )
        self.recurrent = ReluGRUCell(channel_size, hidden_size)
        self.linear_r = nn.Linear(hidden_size, hidden_size)
        self.recurrent_skip = ReluGRUCell(channel_size, hidden_size)
        self.linear_skip = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(skip_size)])
        self.autoregressive = nn.ModuleList([nn.Linear(channel_size, hidden_size) for _ in range(autoregressive_size)])

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self._position_embedding = position_embedding
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size
        self.skip_size = skip_size
        self.autoregressive_size = autoregressive_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        if self._position_embedding:
            inputs = self.emb(inputs)
        inputs = self.convolution(inputs.transpose(1, 2)).transpose(1, 2)
        h_r = torch.zeros(inputs.size(0), self.__hidden_size, device=inputs.device)
        h_s = [torch.zeros(inputs.size(0), self.__hidden_size, device=inputs.device) for _ in range(self.skip_size)]
        hiddens = []
        for t in range(inputs.size(1)):
            h_r = self.recurrent(inputs[:, t, :], h_r)
            h_s[t % self.skip_size] = self.recurrent_skip(inputs[:, t, :], h_s[t % self.skip_size])
            h_d = self.linear_r(h_r)
            for i in range(self.skip_size):
                h_d += self.linear_skip[i](h_s[(t - i) % self.skip_size])
            for i in range(self.autoregressive_size):
                if t >= i:
                    h_d += self.autoregressive[i](inputs[:, t - i, :])
                else:
                    break
            hiddens.append(h_d)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens, hiddens[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module()
class LSTNetAttn(LSTNetSkip):
    def __init__(
        self,
        position_embedding: bool,
        emb_type: str,
        hidden_size: int,
        channel_size: int,
        attention_window: int,
        autoregressive_size: int,
        kernel_size: int = 3,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """The LstNet-Attn network described in http://arxiv.org/abs/1703.07015.

        Args:
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding. Can be static or learn.
            hidden_size: Hidden dimension of the network
            channel_size: The channel size of convolution layer.
            attention_window: The size of attention window of temporal attention layer.
            autoregressive_size: The input window of autoregressive component.
            kernel_size: Kernel size of convolution layer.
        """
        nn.Module.__init__(self)
        self.convolution = nn.Sequential(
            nn.Conv1d(input_size, channel_size, kernel_size, padding=(kernel_size - 1) // 2), nn.ReLU()
        )
        self.recurrent = ReluGRUCell(channel_size, hidden_size)
        self.attention = VanillaAttention()
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.autoregressive = nn.ModuleList([nn.Linear(channel_size, hidden_size) for _ in range(autoregressive_size)])

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self._position_embedding = position_embedding
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size
        self.attention_window = attention_window
        self.autoregressive_size = autoregressive_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        if self._position_embedding:
            inputs = self.emb(inputs)
        inputs = self.convolution(inputs.transpose(1, 2)).transpose(1, 2)
        h_r = torch.zeros(inputs.size(0), self.__hidden_size, device=inputs.device)
        h_rs = []
        hiddens = []
        for t in range(inputs.size(1)):
            h_r = self.recurrent(inputs[:, t, :], h_r)
            h_rs.append(h_r)
            start = max(0, t - self.attention_window + 1)
            key = value = torch.stack(h_rs[start : t + 1], dim=1)
            h_d = self.attention_linear(torch.cat([self.attention(h_r, key, value), h_r], dim=-1))
            for i in range(self.autoregressive_size):
                if t >= i:
                    h_d += self.autoregressive[i](inputs[:, t - i, :])
                else:
                    break
            hiddens.append(h_d)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens, hiddens[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
