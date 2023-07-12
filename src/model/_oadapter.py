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
        image_shape: Tuple[int, int] | None = None,
    ):
        self.image_shape = (200, 200) if image_shape is None else image_shape

        super().__init__(
            output_shape=(math.prod(self.image_shape), num_output_channels)
        )
        self.linear = nn.Linear(num_output_channels, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward Impl.

        :param x: input tensor
        :return: heatmap
        """
        out: Tensor = self.linear(x)
        out = out.permute(0, 2, 1)
        out = out.reshape(x.shape[0], 1, *self.image_shape)

        return {"heatmap": out}


class ClassHeatmapOA(OutputAdapter):
    """"""

    def __init__(
        self,
        class_names: List[str],
        num_output_channels: int,
        image_shape: Tuple[int, int] | None = None,
    ):
        self.image_shape = (200, 200) if image_shape is None else image_shape

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


class OccupancyFlowOA(OutputAdapter):
    """Use two different linear layers to decode occupancy and flow predictions"""

    def __init__(self, num_output_channels: int, image_shape: Tuple[int, int]):
        self.image_shape = (256, 256) if image_shape is None else image_shape
        super().__init__(
            output_shape=(math.prod(self.image_shape), num_output_channels)
        )
        self.occupancy = nn.Linear(num_output_channels, 1)
        self.flow = nn.Linear(num_output_channels, 2)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """"""
        occ = self.occupancy(x).permute(0, 2, 1)
        occ = occ.reshape(x.shape[0], 1, *self.image_shape)

        flow = self.flow(x).permute(0, 2, 1)
        flow = flow.reshape(x.shape[0], 2, *self.image_shape)

        return {"heatmap": occ, "flow": flow}


class OccupancyRefinePre(OutputAdapter):
    """Refine before passing through the final linear layer
    to predict occupancy"""

    def __init__(
        self,
        num_output_channels: int,
        conv_dim: int,
        kernel_size: int | List[int],
        image_shape: Tuple[int, int],
    ):
        super().__init__(output_shape=(math.prod(image_shape), num_output_channels))
        self.image_shape = image_shape

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                num_output_channels,
                conv_dim,
                kernel_size[0],
                padding=kernel_size[0] // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 1, kernel_size[1], padding=kernel_size[1] // 2),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, *self.image_shape)
        out = self.conv(x)
        return {"heatmap": out}


class OccupancyRefinePost(OutputAdapter):
    """Refine after predicting occupancy"""

    def __init__(
        self,
        num_output_channels: int,
        conv_dim: int,
        kernel_size: int | List[int],
        image_shape: Tuple[int, int],
    ):
        super().__init__(output_shape=(math.prod(image_shape), num_output_channels))
        self.image_shape = image_shape
        self.linear = nn.Linear(num_output_channels, 1)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_dim, kernel_size[0], padding=kernel_size[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 1, kernel_size[1], padding=kernel_size[1] // 2),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out: Tensor = self.linear(x)
        out = out.permute(0, 2, 1)
        out = out.reshape(x.shape[0], 1, *self.image_shape)
        out = self.conv(out)
        return {"heatmap": out}
