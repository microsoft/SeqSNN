from typing import List, Optional

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .tsrnn import NETWORKS, TSRNN

import sys
sys.path.append("./forecaster")
from forecaster.module.positional_encoding import PositionEmbedding


class SNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_steps: int = 50,
        grad_slope: float = 25.,
        beta: float = 0.99,
        output_mems: bool = False,
    ):
        super().__init__()
        # self.spike_grad = surrogate.fast_sigmoid(slope=grad_slope)
        self.spike_grad = surrogate.atan(alpha=2.0)
        self.input_size = input_size
        self.num_steps = num_steps
        self.beta = beta
        self.full_rec = output_mems
        self.lif = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=output_mems)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        # print(inputs.size()) # BC, H or BC, H, T
        if inputs.size(-1) == self.input_size:
            # assume static spikes:
            cur = self.linear(inputs)
            static = True
        elif inputs.size(-1) == self.num_steps and inputs.size(-2) == self.input_size:
            # assume dynamic spikes:
            cur = self.linear(inputs.transpose(-1, -2)).transpose(-1, -2) # BC, H, T
            static = False
        else:
            raise ValueError(f"Input size mismatch! Got {inputs.size()} but expected (..., {self.input_size}, {self.num_steps}) or (..., {self.input_size})")
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
            # print(torch.sum(spks).item(), torch.numel(spks), torch.sum(spks) / torch.numel(spks))
            return spks, mems
        else:
            for i_step in range(self.num_steps):
                if static:
                    spk = self.lif(cur)
                else:
                    spk = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
            spks = torch.stack(spk_rec, dim=-1)
            # print(torch.sum(spks).item(), torch.numel(spks), torch.sum(spks) / torch.numel(spks))
            return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2) # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1) # batch, C, L, 1
        enc = self.enc(delta) # batch, C, L, output_size
        enc = enc.permute(0, 3, 1, 2) # batch, output_size, C, L
        spks = self.lif(enc)
        return spks

class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())) # out_size batch L C
        inputs = inputs.permute(1,0,3,2) # batch out_size L C
        spks = self.lif(inputs)
        return spks

class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(alpha=2.0), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, C, L
        enc = self.encoder(inputs) # B, output_size, C, L
        spks = self.lif(enc)
        return spks


@NETWORKS.register_module("SNN")
class TSSNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv"
    ):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(*[
            SNNCell(
                hidden_size,
                hidden_size,
                num_steps=num_steps,
                grad_slope=grad_slope,
                output_mems=(i==layers-1)
            ) for i in range(layers)
        ])

        self.__output_size = hidden_size

    def forward(
        self, 
        inputs: torch.Tensor,
    ):
        # print(inputs.shape) # B, L, C
        for layer in self.net:
            utils.reset(layer)
        hiddens = self.encoder(inputs) # B, L, H
        bs, t, h = hiddens.size() # B, L, H
        for i in range(t):
            spks, mems = self.net(hiddens[:, i, :])
        # print(mems.size()) # B, H, Time Step
        return spks.transpose(1, 2), spks[:, :, -1] # B * Time Step * H, B * H
        # return mems.transpose(1, 2), mems[:, :, -1] # B * Time Step * H, B * H

    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("SNN2d")
class TSSNN2D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        encoder_dim: int = 8,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str="none",
        pe_mode: str="concat", # "add" or concat
        neuron_pe_scale: float=1000.0, # "100" or "1000" or "10000"
    ):
        
        super().__init__()
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.hidden_size = hidden_size
        if encoder_type == "conv":
            self.temporal_encoder = ConvEncoder(encoder_dim)
        elif encoder_type == "delta":
            self.temporal_encoder = DeltaEncoder(encoder_dim)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")
        
        self.pe = PositionEmbedding(pe_type=pe_type, pe_mode=pe_mode, neuron_pe_scale=neuron_pe_scale, input_size=input_size, max_len=max_length, num_pe_neuron=self.num_pe_neuron, dropout=0.1, num_steps=num_steps)
        self.encoder = nn.Linear(encoder_dim, self.hidden_size)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or \
            (self.pe_type == "random" and self.pe_mode == "concat"):
            
            self.__output_size = hidden_size * (input_size + num_pe_neuron)
        else:
            self.__output_size = hidden_size * num_steps

        self.net = nn.Sequential(*[
            SNNCell(
                hidden_size,
                hidden_size,
                num_steps=num_steps,
                grad_slope=grad_slope,
                output_mems=(i==layers-1)
            ) for i in range(layers)
        ])
        
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        # print(inputs.size()) # inputs: B, L, C
        utils.reset(self.temporal_encoder)
        for layer in self.net:
            utils.reset(layer)
        h = self.temporal_encoder(inputs) # B, H_encoder, C, L
        # B, H_encoder, C, L -> B, C, L, H_hidden -> B, H_hidden, C, L
        h = self.encoder(h.permute(0,2,3,1)).permute(0,3,1,2)
        if self.pe_type != "none":
            # B, H, C, L -> H B L C' -> B H C' L
            h = self.pe(h.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        bs, hidden_size, c_num, length = h.size()
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size) # BC, L, H
        # print(h.size()) # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        # print(spks.size()) # BC, H, Time Step
        # print(mems.size()) # BC, H, Time Step
        spks = spks.reshape(bs, c_num * hidden_size, -1) # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1) # B, CH, Time Step
        # return mems.transpose(1, 2), mems[:, :, -1] # B * Time Step * CH, B * CH
        return spks.transpose(1, 2), spks[:, :, -1] # B * Time Step * CH, B * CH
        
    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("ISNN2d")
class ITSSNN2D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        elif encoder_type == "repeat":
            self.encoder = RepeatEncoder(hidden_size) 
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")
        
        self.net = nn.Sequential(*[
            SNNCell(
                hidden_size,
                hidden_size,
                num_steps=num_steps,
                grad_slope=grad_slope,
                output_mems=(i==layers-1)
            ) for i in range(layers)
        ])

        # self.__output_size = hidden_size * input_size
        self.__output_size = hidden_size * num_steps
        
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # C: number of variate (tokens), can also includes covariates

        # print(inputs.size()) # inputs: B, L, C
        utils.reset(self.encoder)
        for layer in self.net:
            utils.reset(layer)
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs) # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size) # BC, L, H
        # print(h.size()) # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        # print(spks.size()) # BC, H, Time Step
        # print(mems.size()) # BC, H, Time Step
        spks = spks.reshape(bs, c_num * hidden_size, -1) # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1) # B, CH, Time Step
        spks = spks.reshape(bs, c_num, -1) # B, C, H*Time Step
        return spks, spks # [B, C, H*Time Step], [B, C, H*Time Step]
        
    @property
    def output_size(self):
        return self.__output_size
