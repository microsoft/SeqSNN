import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask: torch.Tensor,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.mask = mask

    def forward(self, input):
        return self.attention(
            input,
            input,
            input,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=self.mask,
        )[0]


class RelativePositionalEncodingSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_len: int,
        ff_dim: int = 512,
        dropout: float = 0.1,
        batch_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_head = n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.batch_first = batch_first
        self.relative_index = torch.arange(2 * max_len - 1).unfold(0, max_len, 1).flip(0)  # (max_len, max_len - 1)
        self.relative_position_table = nn.parameter.Parameter(torch.zeros(2 * max_len - 1, n_head))
        assert d_model % n_head == 0

    def forward(self, x, **kwargs):
        x = x + self._attention(self.norm1(x))
        x = x + self._ff(self.norm2(x))
        return x

    def _attention(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq, batch, d_model = x.size()
        q, k, v = (
            self.q_linear(x).reshape(seq, batch * self.n_head, -1),
            self.k_linear(x).reshape(seq, batch * self.n_head, -1),
            self.v_linear(x).reshape(seq, batch * self.n_head, -1),
        )  # (seq, batch * self.n_head, d_head)
        weight = torch.einsum("ijk,ljk->jil", q, k).reshape(
            batch, self.n_head, seq, seq
        )  # (batch * self.n_head, seq, seq)
        relative_position_encoding = (
            self.relative_position_table[self.relative_index.view(-1)].view(seq, seq, self.n_head).permute(2, 0, 1)
        )  # (n_head, seq, seq)
        weight = (weight + relative_position_encoding).softmax(dim=-1).reshape(-1, seq, seq)
        attn_outs = torch.einsum("ijk,kil->jil", weight, v).reshape(
            seq, batch, d_model
        )  # (seq, batch * self.n_head, d_head)
        if self.batch_first:
            attn_outs = attn_outs.transpose(0, 1)
        return self.dropout1(attn_outs)

    def _ff(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class ExpSelfAttention(RelativePositionalEncodingSelfAttention):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_len: int,
        alpha: float,
        ff_dim: int = 512,
        dropout: float = 0.1,
        batch_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_head = n_head
        self.alpha = alpha

        self.linear = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.batch_first = batch_first

    def _attention(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq, batch, d_model = x.size()
        x = self.linear(x).reshape(seq, batch, self.n_head, -1)
        weights = [torch.tensor([self.alpha ** (i + 1) for i in range(seq)]).to(x.device).roll(i) for i in range(seq)]
        weights = torch.stack(weights).flip(1).flip(0).tril()
        weights = torch.diag(1 - weights.sum(1)) + weights  # seq, seq
        attn_outs = torch.einsum("hjkl,ih->ijkl", x, weights).reshape(seq, batch, d_model)
        if self.batch_first:
            attn_outs = attn_outs.transpose(0, 1)
        return attn_outs
