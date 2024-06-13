from typing import Optional

from forecaster.module import Chomp1d, Chomp2d
from forecaster.network.tsrnn import NETWORKS, TSRNN
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import sys
sys.path.append("./forecaster")
from forecaster.module.positional_encoding import PositionEmbedding

class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.repeat(tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())) # out_size batch L C
        inputs = inputs.permute(1,0,3,2) # batch out_size L C
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
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(alpha=2.0), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # batch, 1, C, L
        enc = self.encoder(inputs) # batch, output_size, C, L
        spks = self.lif(enc)
        return spks

class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False)
        
    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2) # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1) # batch, C, L, 1
        enc = self.enc(delta) # batch, C, L, output_size
        enc = enc.permute(0, 3, 1, 2) # batch, output_size, C, L
        spks = self.lif(enc)
        return spks

class SpikeTemporalBlock2D(nn.Module):
    def __init__(
            self, 
            n_inputs, 
            n_outputs, 
            kernel_size, 
            stride, 
            dilation, 
            padding,  
            num_steps=4,
            ):
        super(SpikeTemporalBlock2D, self).__init__()
        self.num_steps = num_steps
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs, n_outputs, (1, kernel_size), stride=stride, padding=(0, padding), dilation=(1, dilation)
            )
        )
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.chomp1 = Chomp2d(padding)
        self.lif1 = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(alpha=2.0), init_hidden=True, threshold=1.0)

        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs, n_outputs, (1, kernel_size), stride=stride, padding=(0, padding), dilation=(1, dilation)
            )
        )
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.chomp2 = Chomp2d(padding)
        self.lif2 = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(alpha=2.0), init_hidden=True, threshold=1.0)

        self.downsample = nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        self.lif = snn.Leaky(beta=0.99, spike_grad=surrogate.atan(alpha=2.0),  init_hidden=True, threshold=1.0)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print("x.shape ", x.shape) # inputs: B, H, C, L
        out1 = self.chomp1(self.bn1(self.conv1(x)))
        # print("out1.shape ", out1.shape) # out1: B, H', C, L
        length = out1.size(-1)
        spk_rec1 = []
        for i in range(self.num_steps):
            spk = self.lif1(out1)
            spk_rec1.append(spk)
        spks1 = torch.stack(spk_rec1, dim=-1) # spks1: B, H, C, L, T
        # print("spks1.shape ", spks1.shape)
        spks1 = spks1.mean(-1) # spks1: B, H, C, L
        
        out2 = self.chomp2(self.bn2(self.conv2(spks1)))
        spk_rec2 = []
        for i in range(self.num_steps):
            spk = self.lif2(out2)
            spk_rec2.append(spk)
        spks2 = torch.stack(spk_rec2, dim=-1) # spks2: B, H, C, L, T
        # print("spks2.shape ", spks2.shape)
        spks2 = spks2.mean(-1) # spks2: B, H, C, L


        if torch.isnan(spks2).any() or torch.isinf(spks2).any():
            print("illegal value in TemporalBlock2D")

        if self.downsample is None:
            res = x
        else:
            res = self.downsample(x)
        # res = self.lif(spks2 + res)
        spk_rec3 = []
        for i in range(self.num_steps):
            spk = self.lif(spks2 + res)
            # spk = self.lif(res)
            spk_rec3.append(spk)
            
        res = torch.stack(spk_rec3, dim=-1) # res: B, H, C, L, T
        res = res.mean(-1)
            
        # res = torch.stack(spk_rec3, dim=-1).mean(-1) + spks2# res: B, H, C, L
        return res


@NETWORKS.register_module("SNN_TCN2D")
class SpikeTemporalConvNet2D(TSRNN):
    def __init__(
        self,
        num_levels: int,
        channel: int,
        dilation: int,
        stride: int = 1,
        num_steps: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str="none",
        pe_mode: str="concat", # "add" or "concat"
        neuron_pe_scale: float=1000.0, # "100" or "1000" or "10000"
    ):
        """
        Args:
            num_channels: The number of convolutional channels in each layer.
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        nn.Module.__init__(self)
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        # self.encoder = ConvEncoder(hidden_size)
        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        elif encoder_type == "repeat":
            self.encoder = RepeatEncoder(hidden_size) 
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")
        
        self.num_steps = num_steps
        self.pe = PositionEmbedding(pe_type=pe_type, pe_mode=pe_mode, neuron_pe_scale=neuron_pe_scale, input_size=input_size, max_len=max_length, num_pe_neuron=self.num_pe_neuron, dropout=0.1, num_steps=num_steps)
        layers = []
        num_channels = [channel] * num_levels
        num_channels.append(1)
        for i in range(num_levels + 1):
            dilation_size = dilation**i
            in_channels = hidden_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                SpikeTemporalBlock2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    num_steps = num_steps
                )
            ]

        self.network = nn.Sequential(*layers)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or \
            (self.pe_type == "random" and self.pe_mode == "concat"):
            self.__output_size = input_size + num_pe_neuron
        else:
            self.__output_size = input_size

    def forward(self, inputs: torch.Tensor):
        # print("inputs.shape ", inputs.shape) # inputs: B, L, C
        B, L, C = inputs.size()
        utils.reset(self.encoder)
        for layer in self.network:
            utils.reset(layer)

        inputs = self.encoder(inputs) # B, H, C, L
        # print("inputs.shape ", inputs.shape) # B, H, C, L
        if self.pe_type != "none":
            # B, H, C, L -> H B L C' -> B H C' L
            inputs = self.pe(inputs.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
        # print("inputs.shape ", inputs.shape) # B, H, C', L
        spks = self.network(inputs)
        # print("spks.shape: ", spks.shape)  # B, 1, C', L
        spks = spks.squeeze(1) # B, C', L
        return spks, spks[:, :, -1] # [B, C', L], [B, C']

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
        