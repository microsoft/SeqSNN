from typing import Optional

from pathlib import Path
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./forecaster")
from spikingjelly.activation_based import surrogate, neuron, functional
import numpy as np
from forecaster.network.tsrnn import NETWORKS, TSRNN
import copy
import math
from forecaster.module.positional_encoding import PositionEmbedding

tau = 2.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True

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
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, D, L
        enc = self.encoder(inputs) # B, T, D, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, D, L
        spks = self.lif(enc) # T, B, D, L
        return spks

class SSA(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.attn_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr/2, backend=backend)

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x):
        # x = x.transpose(0, 1)

        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1) # TB L D
        q_m_out = self.q_m(x_for_qkv) # TB L D
        q_m_out = self.q_bn(q_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_bn(k_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_bn(v_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x

class MLP(nn.Module):
    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None):
        super().__init__()
        # self.length = length
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.flatten(0, 1) # TB L D
        x = self.fc1(x) # TB L H
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, self.hidden_features).contiguous()
        x = self.lif1(x)
        x = x.flatten(0, 1) # TB L H
        x = self.fc2(x) # TB L D
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        x = self.lif2(x)
        return x

class Block(nn.Module):
    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = SSA(length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = MLP(length=length, tau=tau, common_thr=common_thr, in_features=dim, hidden_features=d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

@NETWORKS.register_module("Spikformer")
class Spikformer(nn.Module):
    def __init__(
            self,
            dim: int,
            d_ff: Optional[int] = None,
            num_pe_neuron: int = 10,
            pe_type: str="none",
            pe_mode: str="concat", # "add" or concat
            neuron_pe_scale: float=1000.0, # "100" or "1000" or "10000"
            depths: int = 2, 
            common_thr: float = 1.0, 
            max_length: int = 5000,
            num_steps: int = 4, 
            heads: int =8, 
            qkv_bias: bool=False, 
            qk_scale: float = 0.125,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None,
            ):
        super(Spikformer, self).__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron

        self.temporal_encoder = ConvEncoder(num_steps)
        self.pe = PositionEmbedding(pe_type=pe_type, pe_mode=pe_mode, neuron_pe_scale=neuron_pe_scale, input_size=input_size, max_len=max_length, num_pe_neuron=self.num_pe_neuron, dropout=0.1, num_steps=num_steps)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or \
            (self.pe_type == "random" and self.pe_mode == "concat"):
            self.encoder = nn.Linear(input_size + num_pe_neuron, dim)
        else:
            self.encoder = nn.Linear(input_size, dim)
        self.init_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.blocks = nn.ModuleList([Block(
            length=max_length, tau=tau, common_thr=common_thr, dim=dim, d_ff=self.d_ff, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale
        ) for _ in range(depths)])
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        functional.reset_net(self)

        x = self.temporal_encoder(x) # B L C -> T B C L
        x = x.transpose(-2, -1) # T B L C
        if self.pe_type != "none":
            x = self.pe(x) # T B L C'
        T, B, L, C = x.shape
        
        x = self.encoder(x.flatten(0, 1)).reshape(T, B, L, -1) # T B L D
        x = self.init_lif(x)

        D = x.shape[-1]

        for i, blk in enumerate(self.blocks):
            x = blk(x) # T B L D
        # print("x.shape: ", x.shape)
        out = x.mean(0)
        return out, out.mean(dim=1) # B L D, B D
    
    @property
    def output_size(self):
        return self.dim
    
    @property
    def hidden_size(self):
        return self.dim