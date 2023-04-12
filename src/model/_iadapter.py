"""Input adapters for Perciever IO
"""

import math
import enum
from typing import List, Optional, Tuple

import einops
import torch
from torch import nn, Tensor


def _debug_plot(tensor: Tensor, figname: str) -> None:
    """Simple function to call when debugging"""
    import cv2
    import numpy as np

    im = tensor.clone().detach().cpu().numpy()
    im_norm = np.zeros_like(im, dtype=np.uint8)
    cv2.normalize(im, im_norm, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(figname, im_norm)


def _generate_positions_for_encoding(spatial_shape, v_min=-1.0, v_max=1.0):
    """
    Create evenly spaced position coordinates for
    spatial_shape with values in [v_min, v_max].

    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    coords = [torch.linspace(v_min, v_max, steps=s) for s in spatial_shape]
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))


def _generate_position_encodings(
    p: Tensor,
    num_frequency_bands: int,
    max_frequencies: Optional[Tuple[int, ...]] = None,
    include_positions: bool = True,
) -> Tensor:
    """Fourier-encode positions p using num_frequency_bands.

    :param p: positions of shape (*d, c) where c = len(d).
    :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
           2-tuple for images, ...). If `None` values are derived from shape of p.
    :param include_positions: whether to include input positions p in returned encodings tensor.
    :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
    """
    encodings = []

    if max_frequencies is None:
        max_frequencies = p.shape[:-1]

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device)
        for max_freq in max_frequencies
    ]
    frequency_grids = []

    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    if include_positions:
        encodings.append(p)

    encodings.extend(
        [torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )
    encodings.extend(
        [torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )

    return torch.cat(encodings, dim=-1)


def _sample_frequency_band(
    p: Tensor,
    num_frequency_bands: int,
    max_frequencies: Optional[Tuple[int, ...]] = None,
    include_positions: bool = True,
) -> Tensor:
    """
    Samples fourier encoding at a relative position coded such that
    three dimensions can be used for spatio-temporal vehicle input
    """

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device)
        for max_freq in max_frequencies
    ]

    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    encodings = [p] if include_positions else []

    encodings.extend(
        [torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )
    encodings.extend(
        [torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )

    return torch.cat(encodings, dim=-1)


class InputAdapter(nn.Module):
    """blah."""

    def __init__(self, num_input_channels):
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x):
        raise NotImplementedError()


class ImageIA(InputAdapter):
    """_summary_

    :param InputAdapter: _description_
    :type InputAdapter: _type_
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...],
        num_frequency_bands: int,
        patchify: int = 1,
        conv_1x1: int = None,
        in_channels: int = 3,
    ):
        num_image_channels, *self.spatial_shape = image_shape
        self.image_shape = tuple(image_shape)
        self.num_frequency_bands = num_frequency_bands
        self.patch_size = patchify  # basically no patching if 1

        self.conv_1x1 = None

        if patchify > 1:
            self.spatial_shape = [s // self.patch_size for s in self.spatial_shape]

        if conv_1x1 is not None:
            num_image_channels = conv_1x1

        super().__init__(
            num_input_channels=num_image_channels
            + self._num_position_encoding_channels()
        )

        if conv_1x1 is not None:
            if patchify > 1:
                in_channels *= patchify**2
            self.conv_1x1 = nn.Conv2d(in_channels, conv_1x1, 1)

        # create encodings for single example
        pos = _generate_positions_for_encoding(self.spatial_shape)
        enc = _generate_position_encodings(pos, self.num_frequency_bands)

        # flatten encodings along spatial dimensions
        enc = einops.rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.spatial_shape) * (
            2 * self.num_frequency_bands + include_positions
        )

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(
                f"Input image shape {tuple(d)} different from required shape {self.image_shape}"
            )

        if self.patch_size > 1:
            x = einops.rearrange(
                x,
                "b c (h dh) (w dw) -> b (dh dw c) h w",
                dh=self.patch_size,
                dw=self.patch_size,
            )

        if self.conv_1x1 is not None:
            x = self.conv_1x1(x)

        x = einops.rearrange(x, "b c ... -> b (...) c")

        # repeat position encoding along batch dimension
        x_enc = einops.repeat(self.position_encoding, "... -> b ...", b=b)
        return torch.cat([x, x_enc], dim=-1)


class TextIA(InputAdapter):
    """blah"""

    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        self.text_embedding = nn.Embedding(vocab_size, num_input_channels)
        self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, num_input_channels))

        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        self.text_embedding.weight.data.uniform_(-0.1, 0.1)
        self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
        b, l = x.shape  # noqa: E741

        # repeat position encodings along batch dimension
        p_enc = einops.repeat(self.pos_encoding[:l], "... -> b ...", b=b)

        return self.text_embedding(x) * self.scale + p_enc


class TrafficIA(InputAdapter):
    """
    Input adapter that tokenizes traffic inputs
    transform_t: type of transform applied on the input for perciever
    class_emb: learnable class embeddings added to input
    map_max: largest map size in dataset, input data will get normalised to this

    Waymo Open Motion Classes: [-1 Invalid, 0 Unset, 1 Vehicle, 2 Pedestrian, 3 Cyclist, 3 None]

    """

    class ClassMode(enum.Enum):
        NONE = enum.auto()
        ONEHOT = enum.auto()
        SEPARATE = enum.auto()

    class InputMode(enum.Enum):
        RAW = enum.auto()
        FPOS = enum.auto()
        FPOS_EXTRA = enum.auto()

    def __init__(
        self,
        input_mode: str,
        map_freq: int = 200,
        yaw_freq: int = 16,
        num_frequency_bands: int = 32,
        class_onehot: bool = False,
        class_names: List[str] | None = None,
    ):
        self.input_mode = TrafficIA.InputMode[input_mode.upper()]

        if class_onehot:
            self.class_mode = TrafficIA.ClassMode.ONEHOT
        elif class_names is not None:
            self.class_mode = TrafficIA.ClassMode.SEPARATE
        else:
            self.class_mode = TrafficIA.ClassMode.NONE

        self.class_names = class_names

        # raw = input data x,y,t,dx,dt,dt,l,h
        num_input_channels = {
            TrafficIA.InputMode.RAW: 8,
            TrafficIA.InputMode.FPOS: num_frequency_bands * 3 * 2,
            TrafficIA.InputMode.FPOS_EXTRA: num_frequency_bands * 3 * 2 + 5,
        }[self.input_mode]

        if self.class_mode == TrafficIA.ClassMode.ONEHOT:
            num_input_channels += 3  # 3 classes

        self.yaw_freq = yaw_freq
        self.map_freq = map_freq
        self.num_frequency_bands = num_frequency_bands
        super().__init__(num_input_channels)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Pad mask is true for masked values for pytorch, we want the opposite in IA"""
        if self.input_mode == TrafficIA.InputMode.RAW:
            return x

        enc_x = _sample_frequency_band(
            x,
            num_frequency_bands=self.num_frequency_bands,
            max_frequencies=[self.map_freq, self.map_freq, self.yaw_freq],
            include_positions=False,
        )
        if self.input_mode == TrafficIA.InputMode.FPOS_EXTRA:
            enc_x = torch.cat([enc_x, x[..., 3:8]], dim=-1)

        if self.class_mode == TrafficIA.ClassMode.ONEHOT:
            onehot = torch.zeros([*enc_x.shape[0:2], 6], device=enc_x.device)
            onehot.scatter_(
                2, x[..., [-1]].to(torch.int64) + 1, torch.ones_like(x[..., [-1]])
            )
            enc_x = torch.cat([enc_x, onehot[..., 2:-1]], dim=-1)

            # ensure that at least one dummy token isn't masked to prevent NaN's
            enc_x = torch.cat([enc_x, torch.zeros_like(enc_x[:, [0], :])], dim=1)
            if pad_mask is not None:
                pad_mask = torch.cat(
                    [pad_mask, torch.zeros_like(pad_mask[:, [0]])], dim=1
                )

        elif self.class_mode == TrafficIA.ClassMode.SEPARATE:
            assert self.class_names is not None
            assert pad_mask is not None

            cls_mask = x[..., [-1]].to(torch.int64) + 1
            pad_mask_ = {}
            for idx, cls_name in enumerate(self.class_names, 2):
                pad_mask_[cls_name] = (cls_mask != idx).squeeze(-1) | pad_mask

            # ensure that at least one dummy token isn't masked to prevent NaN's
            enc_x = torch.cat([enc_x, torch.zeros_like(enc_x[:, [0], :])], dim=1)
            for mask in pad_mask_:
                pad_mask_[mask] = torch.cat(
                    [pad_mask_[mask], torch.zeros_like(pad_mask_[mask][:, [0]])],
                    dim=1,
                )

            pad_mask = pad_mask_

        return enc_x, pad_mask


class SignalIA(InputAdapter):
    """
    Input Adapter for traffic signals
    Notes for OneHot: zero state is unknown, ignore labels are -1, therefore we create enough space
    for 10 in the one hot, and then slice off the first two
    """

    class InputMode(enum.Enum):
        RAW = enum.auto()
        FPOS = enum.auto()

    class ClassMode(enum.Enum):
        NONE = enum.auto()
        ONEHOT = enum.auto()
        SCALAR = enum.auto()

    def __init__(
        self,
        input_mode: str | InputMode,
        map_freq: int = 200,
        num_frequency_bands: int = 32,
        max_enum: int = 8,
        onehot: bool = False,
        int_cls: bool = False,
    ):
        if onehot:
            self.class_mode = SignalIA.ClassMode.ONEHOT
            num_input_channels = max_enum
        elif int_cls:
            self.class_mode = SignalIA.ClassMode.SCALAR
            num_input_channels = 1
        else:
            self.class_mode = SignalIA.ClassMode.NONE
            num_input_channels = 0

        self.input_mode = (
            SignalIA.InputMode[input_mode.upper()]
            if isinstance(input_mode, str)
            else input_mode
        )

        self.map_freq = map_freq
        self.num_frequency_bands = num_frequency_bands
        self.max_enum = max_enum
        num_input_channels += {
            SignalIA.InputMode.RAW: 2,
            SignalIA.InputMode.FPOS: num_frequency_bands * 4,
        }[self.input_mode]

        super().__init__(num_input_channels)

    def forward(
        self, x: Tensor, pad_mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor | None]:
        """X input tensor is [x,y,state]"""
        match self.input_mode:
            case SignalIA.InputMode.FPOS:
                enc_x = _sample_frequency_band(
                    x,
                    num_frequency_bands=self.num_frequency_bands,
                    max_frequencies=(self.map_freq, self.map_freq),
                    include_positions=False,
                )
            case SignalIA.InputMode.RAW:
                enc_x = x[..., 0:2]  # Only grab x,y

        match self.class_mode:
            case SignalIA.ClassMode.ONEHOT:
                # n classes = max_enum + 2 since -1 is invalid and 0 is none
                onehot = torch.zeros(
                    [*enc_x.shape[0:2], self.max_enum + 2], device=enc_x.device
                )
                # add 1 to class values to shift from -1...max_enum to 0...max_enum + 1
                onehot.scatter_(
                    2, x[..., [-1]].to(torch.int64) + 1, torch.ones_like(x[..., [-1]])
                )
                # skip [0,1] which is [invalid, none]
                enc_x = torch.cat([enc_x, onehot[..., 2:]], dim=-1)
            case SignalIA.ClassMode.SCALAR:
                enc_x = torch.cat([enc_x, x[..., [-1]]], dim=-1)
            case SignalIA.ClassMode.NONE:
                pass

        # ensure that at least one dummy token isn't masked to prevent NaN's
        if pad_mask is not None:
            enc_x = torch.cat([enc_x, torch.zeros_like(enc_x[:, [0], :])], dim=1)
            pad_mask = torch.cat([pad_mask, torch.zeros_like(pad_mask[:, [0]])], dim=1)

        return enc_x, pad_mask
