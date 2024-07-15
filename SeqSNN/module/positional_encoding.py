import math
import copy

from torch import nn
import torch
from spikingjelly.activation_based import surrogate, neuron


def generate_ones_and_minus_ones_matrix(rows, cols):
    random_matrix = torch.randint(0, 2, (rows, cols))
    binary_matrix = torch.where(
        random_matrix == 0,
        -1 * torch.ones_like(random_matrix),
        torch.ones_like(random_matrix),
    )
    return binary_matrix.float()


class RandomPE(nn.Module):
    def __init__(
        self,
        d_model,
        pe_mode="concat",
        num_pe_neuron=10,
        neuron_pe_scale=1000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.max_len = 5000  # different from windows
        self.pe_mode = pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = generate_ones_and_minus_ones_matrix(
            self.max_len, self.num_pe_neuron
        )  # MaxL, Neur
        pe = pe.unsqueeze(0).transpose(0, 1)  # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # # L, TB, D
        # if self.pe_mode == "concat":
        #     x = torch.concat([x, self.pe[: x.size(0), :].repeat(1, x.size(1), 1)], dim=-1)
        #     # print(x.shape) # L TB D'
        # elif self.pe_mode == "add":
        #     x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)

        # T, B, L, D
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3)  # B, T, L, D
        x = x.flatten(1, 2)  # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[: x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            x = x + self.pe[: x.size(-2), :].transpose(0, 1)
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1)  # TL, B D
        x = x.reshape(T, L, B, -1)  # T, L, B, D
        x = x.permute(0, 2, 1, 3)  # T, B, L, D
        return self.dropout(x)


class NeuronPE(nn.Module):
    def __init__(
        self,
        d_model,
        pe_mode="concat",
        num_pe_neuron=10,
        neuron_pe_scale=10000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.max_len = 50000  # different from windows
        self.pe_mode = pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = torch.zeros(self.max_len, self.num_pe_neuron)  # MaxL, Neur
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(
            1
        )  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, self.num_pe_neuron, 2).float()
            * (-math.log(neuron_pe_scale) / self.num_pe_neuron)
        )
        div_term_single = torch.exp(
            torch.arange(0, self.num_pe_neuron - 1, 2).float()
            * (-math.log(neuron_pe_scale) / self.num_pe_neuron)
        )
        pe[:, 0::2] = torch.heaviside(
            torch.sin(position * div_term) - 0.8, torch.tensor([1.0])
        )
        pe[:, 1::2] = torch.heaviside(
            torch.cos(position * div_term_single) - 0.8, torch.tensor([1.0])
        )
        pe = pe.unsqueeze(0).transpose(0, 1)  # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # # L, TB, D
        # if self.pe_mode == "concat":
        #     x = torch.concat([x, self.pe[: x.size(0), :].repeat(1, x.size(1), 1)], dim=-1)
        #     # print(x.shape) # L TB D'
        # elif self.pe_mode == "add":
        #     x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)

        # T, B, L, D
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3)  # B, T, L, D
        x = x.flatten(1, 2)  # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[: x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            # print(self.pe[:x.size(-2), :].shape)
            x = x + self.pe[: x.size(-2), :].transpose(0, 1)
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1)  # TL, B D
        x = x.reshape(T, L, B, -1)  # T, L, B, D
        x = x.permute(0, 2, 1, 3)  # T, B, L, D
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
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # MaxL, D
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # MaxL, 1
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        div_term_single = torch.exp(
            torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: L, TB, D
        x = x + self.pe[: x.size(0), :]
        x = self.dropout(x)
        return x


class ConvPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_steps=4):

        super().__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(
            step_mode="m",
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: L, TB, D
        L, TB, D = x.shape
        x_feat = x.permute(1, 2, 0)  # TB, D, L
        x_feat = self.rpe_conv(x_feat)  # TB, D, L
        x_feat = (
            self.rpe_bn(x_feat).reshape(self.T, int(TB / self.T), D, L).contiguous()
        )  # T, B, D, L
        x_feat = self.rpe_lif(x_feat)
        x_feat = x_feat.flatten(0, 1)  # TB, D, L
        x_feat = self.dropout(x_feat)  # TB, D, L
        x_feat = x_feat.permute(2, 0, 1)  # L, TB, D
        x = x + x_feat
        return x


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        pe_type: str,
        max_len: int = 5000,
        pe_mode: str = "add",
        num_pe_neuron: int = 10,
        neuron_pe_scale: float = 1000.0,
        dropout=0.1,
        num_steps=4,
    ):
        super().__init__()
        self.emb_type = pe_type
        if pe_type in ["learn", "none"]:
            self.emb = nn.Embedding(max_len, input_size)
        elif pe_type == "conv":
            self.emb = ConvPE(
                d_model=input_size,
                max_len=max_len,
                dropout=dropout,
                num_steps=num_steps,
            )
        elif pe_type == "static":
            self.emb = StaticPE(d_model=input_size, max_len=max_len, dropout=dropout)
        elif pe_type == "neuron":
            self.emb = NeuronPE(
                d_model=input_size,
                pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                dropout=dropout,
                num_steps=num_steps,
            )
        elif pe_type == "random":
            self.emb = RandomPE(
                d_model=input_size,
                pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                dropout=dropout,
                num_steps=num_steps,
            )
        else:
            raise ValueError("Unknown embedding type: {}".format(pe_type))

    def forward(self, x):
        if self.emb_type == "learn":
            # T, B, L, D = x.shape # x: T, B, L, D
            # x = x.flatten(0, 1) # TB, L, D
            tmp = torch.arange(
                end=x.size()[1], device=x.device
            )  # [0,1,2,...,L-1], shape: L
            embedding = self.emb(tmp)  # shape: L, D
            embedding = embedding.repeat([x.size()[0], 1, 1])  # TB, L, D'
            x = x + embedding
            # x = x.reshape(T, B, L, -1)
        elif self.emb_type in ["static", "conv"]:
            T, B, L, _ = x.shape  # x: T, B, L, D
            x = x.flatten(0, 1)  # TB, L, D
            x = self.emb(x.transpose(0, 1)).transpose(0, 1)  # x: TB, L, D'
            x = x.reshape(T, B, L, -1)
        elif self.emb_type in ["neuron", "random"]:
            T, B, L, _ = x.shape  # x: T, B, L, D
            # T, B, L, D
            x = self.emb(x)
            x = x.reshape(T, B, L, -1)
        return x  # T, B, L, D'
