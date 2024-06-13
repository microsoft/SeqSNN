from typing import Optional

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
# print(torch.__version__)
import sys
sys.path.append("./forecaster")
from spikingjelly.activation_based import surrogate, neuron, functional
import numpy as np
from forecaster.network.tsrnn import NETWORKS, TSRNN

tau = 2.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True
# common_thr = 1.0
# attn_thr = common_thr / 4

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

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding_inverted, self).__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = neuron.LIFNode(tau = tau, step_mode="m", detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x: T B L C
        # print("x.shape: ", x.shape)
        T,B,L,C = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0,1) # TB C L
        x = self.value_embedding(x) # TB C H
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2) # TB C H
        x = x.reshape(T, B, C, self.d_model)
        x = self.lif(x) # T B C H
        return x

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

        T, B, C, H = x.shape
        x_for_qkv = x.flatten(0, 1) # TB C H

        q_m_out = self.q_m(x_for_qkv) # TB C H
        q_m_out = self.q_bn(q_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, H).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, C, self.heads, H // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_bn(k_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, H).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, C, self.heads, H // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_bn(v_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, H).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, C, self.heads, H // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        # print(attn.shape)
        x = attn @ v  # x_shape: T * B * heads * C * H//heads
        # print(x.shape)

        x = x.transpose(2, 3).reshape(T, B, C, H).contiguous()
        # print(x.shape)
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, C, H).contiguous())
        return x

class MLP(nn.Module):
    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None, ):
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
        # x = x.transpose(0, 1) # T B C H
        T, B, C, H = x.shape
        x = x.flatten(0, 1)
        x = self.fc1(x)
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, self.hidden_features).contiguous()
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.fc2(x)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, H).contiguous()
        x = self.lif2(x)
        # x = x.transpose(0, 1) # T B C H
        return x

class Block(nn.Module):
    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = SSA(length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = MLP(length=length, tau=tau, common_thr=common_thr, in_features=dim, hidden_features=d_ff)
        # self.conv = conv(d_model=dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


@NETWORKS.register_module("iSpikformer")
class iSpikformer(nn.Module):
    def __init__(
            self,
            dim: int,
            d_ff: Optional[int] = None,
            depths: int = 2, 
            common_thr: float = 1.0, 
            max_length: int = 100,
            num_steps: int = 4, 
            heads: int =8, 
            qkv_bias: bool=False, 
            qk_scale: float = 0.125,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None,
            encoder_type: Optional[str] = "conv",
            ):
        super(iSpikformer, self).__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        if encoder_type == "conv":
            self.encoder = ConvEncoder(num_steps)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(num_steps)
        elif encoder_type == "repeat":
            self.encoder = RepeatEncoder(num_steps) 
        self.emb = DataEmbedding_inverted(max_length, dim)
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
        functional.reset_net(self.encoder)
        functional.reset_net(self.emb)
        functional.reset_net(self.blocks)
        x = self.encoder(x) # B L C -> T B C L
        # print(x.shape)
        # x = x.repeat(tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())) # T B L D
        
        x = x.transpose(2, 3) # T B L C
        
        x = self.emb(x)  # T B C H
        for i, blk in enumerate(self.blocks):
            x = blk(x) # T B C H
        # print("x.shape: ", x.shape)
        out = x[-1, :, :, :]
        # out = x.mean(0)
        return out, out # B C H, B C H
    
    @property
    def output_size(self):
        return self.dim # H
    
    @property
    def hidden_size(self):
        return self.dim