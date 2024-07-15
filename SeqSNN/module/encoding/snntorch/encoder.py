import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # out_size batch L C
        inputs = inputs.permute(1, 0, 3, 2)  # batch out_size L C
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
