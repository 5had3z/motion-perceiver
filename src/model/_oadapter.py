"""Output adapters for Perciever IO
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor


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

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(dim=1)


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
