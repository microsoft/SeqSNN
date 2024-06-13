from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from forecaster.network.tsrnn import NETWORKS
from forecaster.module import PositionEmbedding


@NETWORKS.register_module()
class Transformer2d(nn.Module):
    def __init__(
        self,
        spatial_embed: bool,
        temporal_embed: bool,
        temporal_emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 4,
        num_heads: int = 4,
        is_bidir: bool = True,
        use_last: bool = False,
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(1, hidden_size), nn.LeakyReLU())
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads, dropout=dropout)
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers)
        if spatial_embed:
            self.spatial_embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.spatial_embedding = None
        if temporal_embed:
            self.temporal_embedding = PositionEmbedding(temporal_emb_type, hidden_size, max_length)
        else:
            self.temporal_embedding = None
        self.out_layer = nn.Sequential(nn.Linear(hidden_size, 1), nn.LeakyReLU())

        self.__output_size = input_size
        self.__hidden_size = hidden_size
        self.is_bidir = is_bidir
        self.use_last = use_last

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size

    def forward(self, inputs):
        bs, temporal_dim, spatial_dim = inputs.size()
        inputs = self.encoder(inputs.view(bs, -1, 1))  # bs, temporal_dim * spatial_dim, 1
        if self.spatial_embedding is not None:
            spatial_embedding = self.spatial_embedding(
                torch.arange(end=spatial_dim, device=inputs.device)
            )  # spatial_dim, emb_dim
            spatial_embedding = spatial_embedding.repeat(bs, temporal_dim, 1)
            # inputs = torch.cat([inputs, spatial_embedding], dim=-1)
            inputs = inputs + spatial_embedding
        if self.temporal_embedding is not None:
            temporal_embedding = self.temporal_embedding(
                torch.zeros(bs, temporal_dim, self.hidden_size, device=inputs.device)
            ).repeat_interleave(spatial_dim, dim=1)
            # inputs = torch.cat([inputs, temporal_embedding], dim=-1)
            inputs = inputs + temporal_embedding
        z = (
            inputs.reshape(bs, temporal_dim, spatial_dim, self.hidden_size)
            .transpose(1, 2)
            .reshape(bs * spatial_dim, temporal_dim, self.hidden_size)
        )

        # temporal encoder
        attn_outs = self.temporal_encoder(z.transpose(0, 1)).transpose(0, 1)
        seq_outs = self.out_layer(attn_outs.reshape(bs, spatial_dim, temporal_dim, -1)).squeeze(-1)
        if self.use_last:
            out = seq_outs[:, :, -1]
        else:
            out = seq_outs.mean(dim=2)
        return seq_outs.transpose(1, 2), out
