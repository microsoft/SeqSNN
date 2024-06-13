from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsrnn import NETWORKS, TSRNN
from ..module import PositionEmbedding


class InputAttention(nn.Module):
    def __init__(self, hidden_size: int, max_length: int):
        super().__init__()
        self.key_net = nn.Linear(hidden_size * 2, max_length)
        self.query_net = nn.Linear(max_length, max_length)
        self.weight_net = nn.Sequential(nn.Tanh(), nn.Linear(max_length, 1))

    def forward(self, query: torch.Tensor, key: torch.Tensor, time: int) -> torch.Tensor:
        weight = self.weight_net(self.key_net(key).unsqueeze(1) + self.query_net(query.transpose(1, 2)))  # B, D, 1
        weight = F.softmax(weight.squeeze(dim=2), dim=1)  # B, D
        return query[:, time, :] * weight


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, input_size: int, max_length: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.attention = InputAttention(hidden_size, max_length)
        self._hidden_size = hidden_size
        self._max_length = max_length

    def forward(self, inputs):
        # inputs: B, T, D
        assert inputs.size()[1] == self._max_length
        hiddens = []
        hidden = torch.zeros(inputs.size()[0], self._hidden_size).to(inputs.device)
        cell = torch.zeros(inputs.size()[0], self._hidden_size).to(inputs.device)
        for i in range(inputs.size()[1]):
            input = self.attention(inputs, torch.cat([hidden, cell], dim=-1), i)
            hidden, cell = self.cell(input, (hidden, cell))
            hiddens.append(hidden)
        return torch.stack(hiddens, dim=1)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, context_size: int):
        super().__init__()
        self.key_net = nn.Linear(hidden_size * 2, context_size)
        self.query_net = nn.Linear(context_size, context_size)
        self.weight_net = nn.Sequential(nn.Tanh(), nn.Linear(context_size, 1))

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        weight = self.weight_net(self.key_net(key).unsqueeze(1) + self.query_net(query))  # B, T, 1
        weight = F.softmax(weight, dim=1)  # B, T, 1
        return (query * weight).sum(dim=1)


class Decoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.attention = TemporalAttention(hidden_size, hidden_size)
        self._hidden_size = hidden_size

    def forward(self, inputs):
        # inputs: B, T, H
        hiddens = []
        contexts = []
        hidden = torch.zeros(inputs.size()[0], self._hidden_size).to(inputs.device)
        cell = torch.zeros(inputs.size()[0], self._hidden_size).to(inputs.device)
        for _ in range(inputs.size()[1]):
            input = self.attention(inputs, torch.cat([hidden, cell], dim=-1))
            contexts.append(input)
            hidden, cell = self.cell(input, (hidden, cell))
            hiddens.append(hidden)
        return torch.cat((torch.stack(hiddens, dim=1), torch.stack(contexts, dim=1)), dim=-1)  # B, T, 2H


@NETWORKS.register_module()
class DARNN(TSRNN):
    def __init__(
        self,
        position_embedding: bool,
        emb_type: str,
        hidden_size: int,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """The DA-RNN network described in http://arxiv.org/abs/1704.02971.

        Args:
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding. Can be static or learn.
            hidden_size: Hidden dimension of the network
        """
        nn.Module.__init__(self)
        self.encoder = Encoder(hidden_size, input_size, max_length)
        self.decoder = Decoder(hidden_size)

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self._position_embedding = position_embedding
        self.__output_size = hidden_size * 2
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        if self._position_embedding:
            inputs = self.emb(inputs)
        hiddens = self.encoder(inputs)
        outputs = self.decoder(hiddens)
        return outputs, outputs[:, -1, :]

    def get_cpc_repre(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
