from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import copy

from .tsrnn import NETWORKS, TSRNN
from ..module import RelativePositionalEncodingSelfAttention, ExpSelfAttention

class NeuronPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(NeuronPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        num_neuron = copy.deepcopy(d_model)
        position = torch.arange(0, max_len, dtype=torch.float) # [0,1,2,..,MaxL-1]
        # w[i] = 2^(-i) * 2 pi, t is the position index, i is the neuron index
        w = torch.tensor([torch.tensor(2.0) * torch.pi * torch.pow(torch.tensor(2.0), torch.tensor(-i)) for i in range(1, num_neuron+1, 1)]).float()
        b = torch.tensor([torch.rand(1) * torch.tensor(2.0) * torch.pi for _ in range(1, num_neuron+1, 1)]).float()
        # heaviside function: I(sin(w[i]t+b)-0.5)
        position = position.unsqueeze(1) # MaxL, 1
        w = w.unsqueeze(0) # 1, D
        b = b.unsqueeze(0) # 1, D
        tmp = torch.matmul(position, w) # MaxL, D
        pe = torch.heaviside(torch.sin(tmp + b)-0.5, torch.tensor([1.0])) # MaxL, D
        pe = pe.unsqueeze(0).transpose(0, 1) # pe: MaxL, 1, D
        self.register_buffer("pe", pe)

    def forward(self, x):
        # L, B, D
        x = x + self.pe[: x.size(0), :] # pe: L, 1, D
        return self.dropout(x)


class StaticPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(StaticPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # MaxL, D
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # MaxL, 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # D/2, 1
        div_term_single = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: L, B, D
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, emb_type: str, input_size: int, max_len: int = 5000, dropout=0.1):
        super(PositionEmbedding, self).__init__()
        self.emb_type = emb_type
        if emb_type == "learn":
            self.emb = nn.Embedding(max_len, input_size)
        elif emb_type == "static":
            self.emb = StaticPE(input_size, max_len=max_len, dropout=dropout)
        elif emb_type == "neuron":
            self.emb = NeuronPE(input_size, max_len=max_len, dropout=dropout)
        else:
            raise ValueError("Unknown embedding type: {}".format(emb_type))

    def forward(self, x):
        # x: B, L, D
        if self.emb_type == "" or self.emb_type == "learn":
            tmp = torch.arange(end=x.size()[1], device=x.device) # [0,1,2,...,L-1], shape: L
            embedding = self.emb(tmp) # shape: L, D
            embedding = embedding.repeat([x.size()[0], 1, 1]) # B, L, D
            x = x + embedding
        elif self.emb_type == "static" or self.emb_type == "neuron":
            x = self.emb(x.transpose(0, 1)).transpose(0, 1)
        return x # B, L, D



def generate_square_subsequent_mask(sz, prev):
    mask = (torch.triu(torch.ones(sz, sz), -prev) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class BatchFirstTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, batch_first=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x, *args, **kwargs):
        if self.batch_first:
            x = x.transpose(0, 1)
        return super().forward(x, *args, **kwargs).transpose(0, 1)


@NETWORKS.register_module("TSTransformer")
class TSTransformer(TSRNN):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: int = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """
        The Transformer network for time-series prediction.

        Args:
            emb_dim: embedding dimension.
            emb_type: "static" or "learn", static or learnable embedding.
            hidden_size: hidden size of the RNN cell.
            dropout: dropout rate.
            num_layers: number of self-attention layers.
            num_heads: number of self-attention heads.
            is_bidir: whether to use bidirectional transformer (without mask).
            max_length: maximum length of the input sequence.
            input_size: input dimension of the time-series data.
            weight_file: path to the pretrained model.

        Raises:
            ValueError: If `emb_type` is not supported.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = BatchFirstTransformerEncoderLayer(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length, dropout=dropout)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        # positional encoding
        if self.emb_dim > 0:
            inputs = self.emb(inputs)

        # non-regressive encoder
        z = self.encoder(inputs) # B, L, D

        # mask generation
        mask = generate_square_subsequent_mask(z.size()[1], 0)
        if torch.cuda.is_available():
            mask = mask.cuda()
        if self.is_bidir:
            mask = torch.zeros_like(mask)

        # regressive encoder
        attn_outs = self.temporal_encoder(z, mask)

        if self.use_last:
            out = attn_outs[:, -1, :]
        else:
            out = attn_outs.mean(dim=1)

        return attn_outs, out # B L D, B D

    def get_cpc_repre(self, inputs):
        """
        Get the representation of the input sequence for cpc pre-training.
        """
        assert (
            self.is_bidir is False
        ), "Conduct CPC pre-training on bidirectional transformer would cause information leakage."

        # positional encoding
        if self.emb_dim > 0:
            inputs = self.emb(inputs)

        # non-regressive encoder
        z = self.encoder(inputs)

        # mask generation
        mask = generate_square_subsequent_mask(z.size()[1], 0)
        if torch.cuda.is_available():
            mask = mask.cuda()

        # regressive encoder
        attn_outs = self.temporal_encoder(z, mask)

        return z, attn_outs

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module()
class RelativeTSTransformer(TSTransformer):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = RelativePositionalEncodingSelfAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True, max_len=max_length
        )
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module()
class ExpTSTransformer(TSTransformer):
    def __init__(
        self,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        alpha: float = 0.5,
        num_layers: int = 1,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        encoder_layers = ExpSelfAttention(
            hidden_size, num_heads, alpha=alpha, dropout=dropout, batch_first=True, max_len=max_length
        )
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self.is_bidir = is_bidir
        self.use_last = use_last
        self.emb_dim = emb_dim
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
