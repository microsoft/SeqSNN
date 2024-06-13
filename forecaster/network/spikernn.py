from typing import List, Optional
import numpy as np
from pathlib import Path
import copy
import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import sys
sys.path.append("./forecaster")
from spikingjelly.activation_based import surrogate, neuron, functional

from forecaster.module.positional_encoding import PositionEmbedding
from forecaster.network.tsrnn import NETWORKS, TSRNN

tau = 2.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True

class SpikeRNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int
    ):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)
        self.lif = neuron.LIFNode(tau = tau, step_mode="m", detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        # T, B, L, C'
        T, B, L, _ = x.shape
        x = x.flatten(0, 1) # TB, L, C'
        # print(x.shape)
        x = self.linear(x)
        x = x.reshape(T, B, L, -1)
        x = self.lif(x) # T, B, L, C'
        return x

class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = neuron.LIFNode(tau = tau, step_mode="m", detach_reset=detach_reset, surrogate_function=surrogate.ATan())
        
    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.repeat(tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())) # T B L C
        inputs = inputs.permute(0, 1, 3, 2) # T B C L
        spks = self.lif(inputs) # T B C L
        return spks

class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = neuron.LIFNode(tau = tau, step_mode="m", detach_reset=detach_reset, surrogate_function=surrogate.ATan())
        
    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2) # B, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1) # B, C, L, 1
        enc = self.enc(delta) # B, C, L, T
        enc = enc.permute(3, 0, 1, 2) # T, B, C, L
        spks = self.lif(enc)
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
        self.lif = neuron.LIFNode(tau = tau, step_mode="m", detach_reset=detach_reset, surrogate_function=surrogate.ATan())
        
    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, C, L
        enc = self.encoder(inputs) # B, T, C, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, C, L
        spks = self.lif(enc) # T, B, C, L
        return spks


@NETWORKS.register_module("SpikeRNN")
class SpikeRNN(nn.Module):
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
        pe_type: str="none",
        pe_mode: str="concat", # "add" or concat
        neuron_pe_scale: float=1000.0, # "100" or "1000" or "10000"
    ):
        super().__init__()
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.neuron_pe_scale = neuron_pe_scale
        if encoder_type == "conv":
            self.temporal_encoder = ConvEncoder(num_steps)
        elif encoder_type == "delta":
            self.temporal_encoder = DeltaEncoder(num_steps)
        elif encoder_type == "repeat":
            self.temporal_encoder = RepeatEncoder(num_steps) 
            
        self.pe = PositionEmbedding(pe_type=pe_type, pe_mode=pe_mode, neuron_pe_scale=neuron_pe_scale, input_size=input_size, max_len=max_length, num_pe_neuron=self.num_pe_neuron, dropout=0.1, num_steps=num_steps)

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.dim = hidden_size + num_pe_neuron
        else:
            self.dim = hidden_size

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.encoder = nn.Linear(input_size + num_pe_neuron, self.dim)
        else:
            self.encoder = nn.Linear(input_size, self.dim )
        self.init_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=1.0, backend=backend)
        
        self.net = nn.Sequential(*[
            SpikeRNNCell(
                input_size = self.dim,
                output_size = self.dim
            ) for i in range(layers)
        ])

        self.__output_size = self.dim

    def forward(
        self, 
        inputs: torch.Tensor,
    ):
        # print(inputs.shape) # B, L, C
        functional.reset_net(self)
        hiddens = self.temporal_encoder(inputs) # T, B, C, L
        hiddens = hiddens.transpose(-2, -1) # T, B, L, C
        T, B, L, _ = hiddens.size() # T, B, L, D
        if self.pe_type != "none":
            hiddens = self.pe(hiddens) # T B L C'
        hiddens = self.encoder(hiddens.flatten(0, 1)).reshape(T, B, L, -1) # T B L D
        hiddens = self.init_lif(hiddens)
        hiddens = self.net(hiddens) # T, B, L, D
        out = hiddens.mean(0)
        return out, out.mean(dim=1) # B L D, B D

    @property
    def output_size(self):
        return self.__output_size
    
    @property
    def hidden_size(self):
        return self.dim


@NETWORKS.register_module("SpikeRNN2d")
class SpikeRNN2D(nn.Module):
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
        num_pe_neuron: int = 10,
        pe_type: str="none",
        pe_mode: str="concat", # "add" or concat
        neuron_pe_scale: float=1000.0, # "100" or "1000" or "10000"
    ):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")
        
        self.net = nn.Sequential(*[
            SpikeRNNCell(
                hidden_size,
                hidden_size,
                num_steps=num_steps,
                grad_slope=grad_slope,
                output_mems=(i==layers-1)
            ) for i in range(layers)
        ])

        self.__output_size = hidden_size * input_size
        # self.__output_size = hidden_size * num_steps
        
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        # print(inputs.size()) # inputs: B, L, C
        
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
        # return mems.transpose(1, 2), mems[:, :, -1] # B * Time Step * CH, B * CH
        return spks.transpose(1, 2), spks[:, :, -1] # B * Time Step * CH, B * CH
        
    @property
    def output_size(self):
        return self.__output_size
