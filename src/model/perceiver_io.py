"""Perciever IO Model."""

from pathlib import Path
from typing import Any, Dict, Optional


import torch
from torch import nn, Tensor
import einops

from ._iadapter import (
    InputAdapter,
    ImageIA,
    TextIA,
    TrafficIA,
    _generate_position_encodings,
    _generate_positions_for_encoding,
)
from ._oadapter import OutputAdapter, ClassificationOA, HeatmapOA


class Sequential(nn.Sequential):
    """Blah."""

    def forward(self, *inputs):
        for module in self:
            inputs = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        return inputs


class SequentialWAux(nn.Sequential):
    """Blah."""

    def forward(self, *inputs):
        latent, *extras = inputs
        for module in self:
            latent = module(latent, *extras)
        return latent


def mlp(num_channels: int):
    """blah"""
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


def cross_attention_layer(
    num_q_channels: int,
    num_kv_channels: int,
    num_heads: int,
    dropout: float,
    residule_query: bool = True,
):
    """blah"""
    if residule_query:
        layer = Sequential(
            Residual(
                CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout),
                dropout,
            ),
            Residual(mlp(num_q_channels), dropout),
        )
    else:
        layer = Sequential(
            CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout),
            Residual(mlp(num_q_channels), dropout),
        )

    return layer


def self_attention_layer(
    num_channels: int,
    num_heads: int,
    dropout: float,
    with_context: bool = False,
):
    """blah"""
    module_t = SelfAttention if not with_context else SelfAttentionStaticFeatures
    layer = Sequential(
        Residual(module_t(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout),
    )
    return layer


def self_attention_block(
    num_layers: int,
    num_channels: int,
    num_heads: int,
    dropout: float,
    with_context: bool = False,
):
    """blah"""
    layers = [
        self_attention_layer(num_channels, num_heads, dropout, with_context)
        for _ in range(num_layers)
    ]
    return Sequential(*layers) if not with_context else SequentialWAux(*layers)


class Residual(nn.Module):
    """blah"""

    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    """blah"""

    def __init__(
        self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(
            x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask
        )[0]


class CrossAttention(nn.Module):
    """
    Simplified version of cross-attention module described in https://arxiv.org/abs/2103.03206.
    Here, the embedding dimension is determined by the number of query channels (num_q_channels)
    whereas in the paper it can be specified separately. This simplification allows re-use of the
    torch.nn.MultiHeadAttention module whereas a full implementation of the paper would require a
    custom multi-head attention implementation.
    """

    def __init__(
        self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float
    ):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    """Traditional self attention where qkv is derived from input tensor"""

    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels,
            num_kv_channels=num_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttentionStaticFeatures(nn.Module):
    """Self attention module where static features are
    also concatenated as extra keys and values as context features.
    Context features should be same channel width as input.
    """

    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels,
            num_kv_channels=num_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, s, s_mask, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        s = self.norm(s)
        xs = torch.cat([x, s], dim=1)
        ctx_mask = (
            torch.ones(x.shape[:-1], device=x.device) if pad_mask is None else pad_mask
        )
        pad_mask = torch.cat([s_mask, ctx_mask], dim=-1)
        return self.attention(x, xs, pad_mask=pad_mask, attn_mask=attn_mask)


class PerceiverEncoder(nn.Module):
    """
    Generic Perceiver IO encoder.
    """

    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_layers: int = 3,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
    ):
        """
        :param input_adapter: Transforms and position-encodes task-specific input to an encoder input of shape
                              (B, M, C_input) where B is the batch size, M the input sequence length and C_input
                              the number of input channels.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (C_latent).
        :param num_layers: Number of encoder layers. An encoder layer is composed of a cross-attention layer and
                           several self-attention layers (= a self-attention block).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param dropout: Dropout for self- and cross-attention layers and residuals.
        """
        super().__init__()

        self.input_adapter = input_adapter
        self.num_layers = num_layers

        def create_perceiver_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=input_adapter.num_input_channels,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                ),
            )

        self.layer_1 = create_perceiver_layer()

        if num_layers > 1:
            # will be used recurrently depending on num_layers
            self.layer_n = create_perceiver_layer()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        """Forward method for percienver encoding

        :param x: input rgb image tensor
        :param pad_mask: mask if padding, defaults to None
        :return: encoded output
        """
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x, pad_mask)

        # repeat initial latent vector along batch dimension
        x_latent = einops.repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.layer_1(x_latent, x, pad_mask)
        for _ in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class PerceiverDecoder(nn.Module):
    """
    Generic Perceiver IO decoder.
    """

    def __init__(
        self,
        output_adapter: OutputAdapter,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.0,
        position_encoding_limit: float = 1.0,
        residule_query: bool = True,
        position_encoding_type: str = "learnable",
        num_frequency_bands: int | None = None,
        max_frequency: float | None = None,
    ):
        """
        :param output_adapter: Transforms generic decoder output of shape (B, K, C_output)
                               to task-specific output. B is the batch size, K the output
                               sequence length and C_output the number of output channels.
                               (K, C_output) is specified via the output_shape property of
                               the output_adapter.
        :param num_latent_channels: Number of latent channels (C_latent) as produced by a
                                    Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param dropout: Dropout for cross-attention layers and residuals.
        :param position_encoding_limit: the min/max value of the position encoding (0,1]
        :param position_encoding_type: type of positional encoding used for output, either
                                       "learnable" or "fourier"
        :param num_frequency_bands: number of frequency bands used for fourier position enc
        """
        super().__init__()

        self.output_adapter = output_adapter

        self._position_encoding_type = position_encoding_type
        if position_encoding_type == "learnable":
            self.output = nn.Parameter(torch.empty(*output_adapter.output_shape))
            self._init_parameters()
            query_channels = output_adapter.output_shape[-1]
        elif position_encoding_type == "fourier":
            assert (
                num_frequency_bands is not None
            ), "num_frequency_bands required for fourier position encoding"
            pos = _generate_positions_for_encoding(
                output_adapter.image_shape,
                v_min=-position_encoding_limit,
                v_max=position_encoding_limit,
            )
            enc = _generate_position_encodings(
                pos,
                num_frequency_bands,
                None if max_frequency is None else [max_frequency] * 2,
                include_positions=False,
            )
            # flatten encodings along spatial dimensions
            enc = einops.rearrange(enc, "... c -> (...) c")
            self.register_buffer("output", enc, persistent=False)
            query_channels = enc.shape[-1]
        else:
            raise NotImplementedError(f"{position_encoding_type=}")

        self.cross_attention = cross_attention_layer(
            num_q_channels=query_channels,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
            residule_query=residule_query,
        )

    @torch.no_grad()
    def _init_parameters(self):
        if self._position_encoding_type == "learnable":
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def onnx_export(self, path: Path) -> None:
        """Export modules as onnx files"""
        print("Exporting Decoder")
        torch.onnx.export(
            self, torch.randn((1, 128, 256)).cuda(), str(path / "output_decoder.onnx")
        )

    def aux_forward(self, x: Tensor) -> Tensor:
        """"""
        output = einops.repeat(self.output, "... -> b ...", b=x.shape[0])
        output = self.cross_attention(output, x)
        return self.output_adapter(output)

    def forward(self, x: Tensor | Dict[str, Tensor]):
        """forward impl"""
        if isinstance(x, dict):
            outputs = {}
            for cls_name, cls_output in x.items():
                outputs[cls_name] = self.aux_forward(cls_output)
            return outputs

        return self.aux_forward(x)


def _show_latent(latent, logit, gt, idx):
    """Helper function to visualise latent
    space when breakpointing in the code"""
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(latent[idx].cpu())
    plt.savefig(f"latent{idx}.png")
    plt.imshow(logit[idx, 0].cpu())
    plt.savefig(f"pred{idx}.png")
    plt.imshow(gt[idx, 0].cpu())
    plt.savefig(f"truth{idx}.png")
