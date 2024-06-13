from pathlib import Path
from typing import Optional

from forecaster.module import PositionEmbedding, get_cell
from forecaster.network import NETWORKS
import torch
import torch.nn as nn


@NETWORKS.register_module()
class RNN2d(nn.Module):
    def __init__(
        self,
        cell_type: str,
        spatial_embed: bool,
        temporal_embed: bool,
        temporal_emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 1,
        is_bidir: bool = False,
        max_length: int = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """The RNN network for time-series prediction.

        Args:
            cell_type: RNN cell type, e.g. "lstm", "gru", "rnn".
            emb_type: "static" or "learn", static or learnable embedding.
            hidden_size: hidden size of the RNN cell.
            dropout: Dropout rate.
            num_layers: Number of layers of the RNN cell.
            is_bidir: Whether to use bidirectional RNN.
            max_length: Maximum length of the input sequence.
            input_size: Input size of the time-series data.
            weight_file: Path to the pretrained model.

        Raises:
            ValueError: If `cell_type` is not supported.
            ValueError: If `emb_type` is not supported.
        """
        super().__init__()
        Cell = get_cell(cell_type)
        self.encoder = nn.Sequential(nn.Linear(1, hidden_size), nn.LeakyReLU())
        self.regressive_encoder = Cell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidir,
            dropout=dropout,
        )
        self.out_layer = nn.Sequential(nn.Linear(hidden_size * 2 if is_bidir else hidden_size, 1), nn.LeakyReLU())

        if spatial_embed:
            self.spatial_embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.spatial_embedding = None
        if temporal_embed:
            self.temporal_embedding = PositionEmbedding(pe_type=temporal_emb_type, input_size=hidden_size, max_len=max_length)
        else:
            self.temporal_embedding = None

        self.__output_size = input_size
        self.__hidden_size = hidden_size
        self.is_bidir = is_bidir

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

        rnn_outs, _ = self.regressive_encoder(z)
        rnn_outs = self.out_layer(rnn_outs.reshape(bs, spatial_dim, temporal_dim, -1)).squeeze(-1)
        return rnn_outs.transpose(1, 2), rnn_outs[:, :, -1]
