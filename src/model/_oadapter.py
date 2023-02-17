"""Output adapters for Perciever IO
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class OutputAdapter(nn.Module):
    """blah"""

    def __init__(self, output_shape: Tuple[int, ...]):
        super().__init__()
        self._output_shape = output_shape

    @property
    def output_shape(self):
        """_summary_

        :return: _description_
        """
        return self._output_shape

    def forward(self, x):
        raise NotImplementedError()


class ClassificationOA(OutputAdapter):
    """blah"""

    def __init__(
        self,
        num_classes: int,
        num_outputs: int = 1,
        num_output_channels: Optional[int] = None,
    ):
        if num_output_channels is None:
            num_output_channels = num_classes

        super().__init__(output_shape=(num_outputs, num_output_channels))
        self.linear = nn.Linear(num_output_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TextOA(ClassificationOA):
    """blah"""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_output_channels: Optional[int] = None,
    ):
        super().__init__(
            num_classes=vocab_size,
            num_outputs=max_seq_len,
            num_output_channels=num_output_channels,
        )


class SegmentationOA(OutputAdapter):
    """blah"""

    def __init__(
        self,
        num_classes: int,
        image_shape: List[int] = None,
        num_output_channels: Optional[int] = None,
    ):
        if num_output_channels is None:
            num_output_channels = num_classes

        self.image_shape = [512, 1024] if image_shape is None else image_shape
        self.num_classes = num_classes

        super().__init__(output_shape=(math.prod(image_shape), num_output_channels))
        self.linear = nn.Linear(num_output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward Impl.

        :param x: input tensor
        :return: semantic segmentation
        """
        out: torch.Tensor = self.linear(x)
        out = out.permute(0, 2, 1)
        out = out.reshape(x.shape[0], self.num_classes, *self.image_shape)

        return {"seg": out}


class HeatmapOA(OutputAdapter):
    """"""

    def __init__(
        self,
        num_output_channels: int,
        image_shape: Optional[List[int]] = None,
    ):
        self.image_shape = [200, 200] if image_shape is None else image_shape

        super().__init__(
            output_shape=(math.prod(self.image_shape), num_output_channels)
        )
        self.linear = nn.Linear(num_output_channels, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward Impl.

        :param x: input tensor
        :return: heatmap
        """
        out: torch.Tensor = self.linear(x)
        out = out.permute(0, 2, 1)
        out = out.reshape(x.shape[0], 1, *self.image_shape)

        return {"heatmap": out}


class ClassHeatmapOA(OutputAdapter):
    """"""

    def __init__(
        self,
        class_names: List[str],
        num_output_channels: int,
        image_shape: Optional[List[int]] = None,
    ):
        self.image_shape = [200, 200] if image_shape is None else image_shape

        super().__init__(
            output_shape=(math.prod(self.image_shape), num_output_channels)
        )

        self.linear_layers = nn.ModuleDict(
            {name: nn.Linear(num_output_channels, 1) for name in class_names}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward Impl.

        :param x: input tensor
        :return: heatmap
        """
        output = {}
        for name, module in self.linear_layers.items():
            output[name] = (
                module(x).permute(0, 2, 1).reshape(x.shape[0], 1, *self.image_shape)
            )

        return output
