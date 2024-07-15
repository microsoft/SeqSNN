from typing import Optional
from pathlib import Path

from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional
from ..base import NETWORKS
from ...module.spike_encoding import SpikeEncoder
from ...module.spike_attention import Block


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        # x: T B L C
        T, B, _, C = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)  # TB C L
        x = self.value_embedding(x)  # TB C H
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)  # TB C H
        x = x.reshape(T, B, C, self.d_model)
        x = self.lif(x)  # T B C H
        return x


@NETWORKS.register_module("iSpikformer")
class iSpikformer(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 100,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](num_steps)

        self.emb = DataEmbedding_inverted(max_length, dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        functional.reset_net(self.encoder)
        functional.reset_net(self.emb)
        functional.reset_net(self.blocks)
        x = self.encoder(x)  # B L C -> T B C L

        x = x.transpose(2, 3)  # T B L C

        x = self.emb(x)  # T B C H
        for blk in self.blocks:
            x = blk(x)  # T B C H
        out = x[-1, :, :, :]
        return out, out  # B C H, B C H

    @property
    def output_size(self):
        return self.dim  # H

    @property
    def hidden_size(self):
        return self.dim
