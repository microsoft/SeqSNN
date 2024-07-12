from typing import Optional
from math import sqrt
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..base import NETWORKS


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _, E = queries.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


@NETWORKS.register_module("iTransformer")
class ITransformer(nn.Module):
    """
    Code copied and modified from the work of ITransformer[1]:
    Paper link: https://arxiv.org/abs/2310.06625
    Original Github repo: https://github.com/thuml/iTransformer

    [1] Liu, Yong, et al. "iTransformer: Inverted Transformers
    Are Effective for Time Series Forecasting." The Twelfth
    International Conference on Learning Representations.
    """

    def __init__(
        self,
        output_attention: bool = False,
        factor: int = 1,
        embed: str = "fixed",
        freq: str = "h",
        e_layers: int = 2,
        d_ff: int = 2048,
        d_model: int = 512,
        dropout: float = 0.1,
        max_length: int = 100,
        n_heads: int = 8,
        activation: str = "gelu",
        class_strategy: str = "projection",
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        """
        Initialize the ITransformer model.

        Args:
            output_attention (bool, optional): Whether to output attention in the encoder. Defaults to False.
            factor (int, optional): Attention factor. Defaults to 1.
            embed (str, optional): Type of embedding. Options: [timeF, fixed, learned].
                Unused in our implementation. Defaults to "fixed".
            freq (str, optional): Frequency of time series. Options: [s, t, h, d, b, w, m].
                Unused in our implementation. Defaults to "h".
            e_layers (int, optional): Number of encoder layers. Defaults to 2.
            d_ff (int, optional): Dimension of the feedforward network. Defaults to 2048.
            d_model (int, optional): Hidden size. Defaults to 512.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            max_length (int, optional): Maximum length of the sequence. Defaults to 100.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            activation (str, optional): Activation function. Defaults to "gelu".
            class_strategy (str, optional): Class strategy. Options: [projection, average, cls_token].
                Unused in the original repository. Defaults to "projection".
            input_size (Optional[int], optional): Size of the input. Defaults to None.
            weight_file (Optional[Path], optional): Path to the weight file. Defaults to None.
        """
        super().__init__()
        self.input_size = input_size
        self.max_length = max_length
        self.output_attention = output_attention
        self.enc_embedding = DataEmbedding_inverted(
            max_length, d_model, embed, freq, dropout
        )
        self.class_strategy = class_strategy
        self.d_model = d_model
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

    def forecast(self, x_enc):
        # FIXME: x_enc_mark, x_dec_mark in the original paper are not used here
        # because our data provider has already provided timestamp information

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev # B L N

        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x_enc, None
        )  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        return enc_out, enc_out  # B N E

    def forward(self, x_enc, mask=None):
        enc_out = self.forecast(x_enc)
        return enc_out  # remember to add a projection layer in Model.

    @property
    def output_size(self):
        return self.d_model  # E

    @property
    def hidden_size(self):
        return self.d_model
