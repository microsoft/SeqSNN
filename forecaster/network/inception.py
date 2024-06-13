from typing import cast, Union, List, Optional
from pathlib import Path

import torch
import torch.nn as nn


from ..module import Conv1dSamePadding, PositionEmbedding
from .tsrnn import NETWORKS


@NETWORKS.register_module("inception")
class InceptionModel(nn.Module):
    def __init__(
        self,
        position_embedding: bool,
        emb_type: str,
        num_blocks: int,
        out_channels: Union[List[int], int],
        bottleneck_channels: Union[List[int], int],
        kernel_sizes: Union[List[int], int],
        use_residuals: Union[str, List[bool], bool] = "default",
        max_length: Optional[int] = 100,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ) -> None:
        """A PyTorch implementation of the InceptionTime model (https://arxiv.org/abs/1909.04939).
        From https://github.com/TheMrGhostman/InceptionTime-Pytorch

        Args:
            position_embedding: Whether to use position embedding
            emb_type: The type of position embedding to use. Can be "learn" or "static".
            num_blocks: The number of inception blocks to use. One inception block consists of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual connector.
            out_channels: The number of "hidden channels" to use. Can be a list (for each block) or an int, in which case the same value will be applied to each block.
            bottleneck_channels: The number of channels to use for the bottleneck. Can be list or int. If 0, no bottleneck is applied.
            kernel_sizes: The size of the kernels to use for each inception block. Within each block, each of the 3 convolutional layers will have kernel size `[kernel_size // (2 ** i) for i in range(3)]`.
            use_residuals: Whether to use residual connections in inception blocks. Can be list or bool. Default to use every three inception blocks.
        """
        super().__init__()

        channels = [input_size] + cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == "default":
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(
            List[bool], self._expand_to_blocks(cast(Union[bool, List[bool]], use_residuals), num_blocks)
        )

        self.blocks = nn.Sequential(
            *[
                InceptionBlock(
                    input_size=channels[i],
                    out_channels=channels[i + 1],
                    residual=use_residuals[i],
                    bottleneck_channels=bottleneck_channels[i],
                    kernel_size=kernel_sizes[i],
                )
                for i in range(num_blocks)
            ]
        )

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length)

        self._position_embedding = position_embedding
        self.__output_size = channels[-1]
        self.__hidden_size = channels[-1]

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @staticmethod
    def _expand_to_blocks(
        value: Union[int, bool, List[int], List[bool]], num_blocks: int
    ) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, (
                f"Length of inputs lists must be the same as num blocks, "
                f"expected length {num_blocks}, got {len(value)}"
            )
        else:
            value = [value] * num_blocks
        return value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self._position_embedding:
            inputs = self.emb(inputs)
        hiddens = self.blocks(inputs.transpose(1, 2)).transpose(1, 2)
        return hiddens, hiddens.mean(dim=1)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(
        self,
        input_size: int,
        out_channels: int,
        residual: bool,
        stride: int = 1,
        bottleneck_channels: int = 32,
        kernel_size: int = 41,
    ) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(input_size, bottleneck_channels, kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else input_size
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(
            *[
                Conv1dSamePadding(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size_s[i],
                    stride=stride,
                    bias=False,
                )
                for i in range(len(kernel_size_s))
            ]
        )

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(
                *[
                    Conv1dSamePadding(
                        in_channels=input_size, out_channels=out_channels, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x
