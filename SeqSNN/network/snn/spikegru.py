from typing import Optional
from pathlib import Path

from spikingjelly.activation_based import surrogate as sj_surrogate
from snntorch import utils
import snntorch as snn
from snntorch import surrogate
import torch
from torch import nn

from ..base import NETWORKS


class GRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_steps: int = 4,
        grad_slope: float = 25.0,
        beta: float = 0.99,
        output_mems: bool = False,
    ):
        super().__init__()
        self.spike_grad = surrogate.atan(alpha=2.0)
        self.input_size = input_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.beta = beta
        self.full_rec = output_mems
        self.lif = snn.Leaky(
            beta=self.beta,
            spike_grad=self.spike_grad,
            init_hidden=True,
            output=output_mems,
        )
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.surrogate_function1 = sj_surrogate.ATan()

    def forward(self, inputs):
        if inputs.size(-1) == self.input_size:
            # assume static spikes:
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(self.linear_ih(inputs), self.hidden_size, dim=1)
            y_hh = torch.split(self.linear_hh(h), self.hidden_size, dim=1)
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = True
        elif inputs.size(-1) == self.num_steps and inputs.size(-2) == self.input_size:
            inputs = inputs.transpose(-1, -2)  # BC, T, H
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size, self.num_steps],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(
                self.linear_ih(inputs).transpose(-1, -2), self.hidden_size, dim=1
            )
            y_hh = torch.split(
                self.linear_hh(h.transpose(-1, -2)).transpose(-1, -2),
                self.hidden_size,
                dim=1,
            )
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = False
        else:
            raise ValueError(
                f"Input size mismatch!"
                f"Got {inputs.size()} but expected (..., {self.input_size}, {self.num_steps}) or (..., {self.input_size})"
            )

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
            return spks, mems
        else:
            for i_step in range(self.num_steps):
                if static:
                    spk = self.lif(cur)
                else:
                    spk = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
            spks = torch.stack(spk_rec, dim=-1)
            return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
        enc = self.enc(delta)  # batch, C, L, output_size
        enc = enc.permute(0, 3, 1, 2)  # batch, output_size, C, L
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
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L
        enc = self.encoder(inputs)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks


@NETWORKS.register_module("SNNGRU")
class TSSNNGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
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
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(
            *[
                GRUCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        for layer in self.net:
            utils.reset(layer)
        hiddens = self.encoder(inputs)
        _, t, _ = hiddens.size()  # B, L, H
        for i in range(t):
            spks, _ = self.net(hiddens[:, i, :])
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * H, B * H

    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("SNNGRU2d")
class TSSNNGRU2D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
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
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(
            *[
                GRUCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size * input_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        utils.reset(self.encoder)
        for layer in self.net:
            utils.reset(layer)
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * CH, B * CH

    @property
    def output_size(self):
        return self.__output_size


@NETWORKS.register_module("ISNNGRU2d")
class ITSSNNGRU2D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 4,
        grad_slope: float = 25.0,
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
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(
            *[
                GRUCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size * num_steps

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # C: number of variate (tokens), can also includes covariates

        utils.reset(self.encoder)
        for layer in self.net:
            utils.reset(layer)
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, _ = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        spks = spks.reshape(bs, c_num, -1)  # B, C, H*Time Step
        return spks, spks  # [B, C, H*Time Step], [B, C, H*Time Step]

    @property
    def output_size(self):
        return self.__output_size
