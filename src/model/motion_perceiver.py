from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import random
from functools import reduce

import einops
import torch
from torch import nn, Tensor

from ._iadapter import InputAdapter, TrafficIA, ImageIA, SignalIA
from ._oadapter import HeatmapOA, ClassHeatmapOA
from . import perceiver_io as pio


class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        input_indicies: List[int],
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        detach_latent: bool = False,
        prop_layer_norm: bool = False,
        random_input_indicies: int = 0,
    ) -> None:
        """
        random_input_indicies selects a random number of indicies between the min
        and max input indicies list during training.
        """
        super().__init__()

        self.input_adapter = input_adapter
        self.latent_dim = num_latent_channels

        self.input_layer = pio.Sequential(
            pio.cross_attention_layer(
                num_q_channels=num_latent_channels,
                num_kv_channels=input_adapter.num_input_channels,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
            ),
            pio.self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads,
                dropout=dropout,
            ),
        )

        self.propagate_layer = pio.self_attention_block(
            num_layers=num_self_attention_layers_per_block,
            num_channels=num_latent_channels,
            num_heads=num_self_attention_heads,
            dropout=dropout,
        )

        self.input_indicies = set(input_indicies)
        self.random_input_indicies = random_input_indicies

        # detach and clone latent at every output index during
        # training so gradients don't flow all the way back
        self.detach_latent = detach_latent

        self.prop_layer_norm = (
            nn.LayerNorm(num_latent_channels) if prop_layer_norm else None
        )

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        raise NotImplementedError()

    def forward(
        self,
        output_idx: Tensor,
        agents: Tensor,
        agents_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """Given an input of dims [Time Batch Tokens Channels]"""

        # repeat initial latent vector along batch dimension
        x_latent = einops.repeat(self.latent, "... -> b ...", b=agents.shape[1])

        max_idx = int(output_idx.max().item()) + 1
        first_input_idx = min(self.input_indicies)
        out_latent = []

        input_indicies = self.input_indicies
        if self.random_input_indicies > 0 and self.training:
            candidates = list(range(min(input_indicies), max(input_indicies)))
            random.shuffle(candidates)
            input_indicies.update(candidates[: self.random_input_indicies])

        for t_idx in range(first_input_idx, max_idx):
            if t_idx in input_indicies:
                x_step, x_mask = agents[t_idx], agents_mask[t_idx]
                x_adapt, x_mask = self.input_adapter(x_step, x_mask)
                x_latent = self.input_layer(x_latent, x_adapt, x_mask)
            else:
                x_latent = self.propagate_layer(x_latent)

            if t_idx in output_idx:
                out_latent.append(x_latent)
                if self.detach_latent and self.training:
                    x_latent = x_latent.detach().clone()

            if self.prop_layer_norm is not None:
                x_latent = self.prop_layer_norm(x_latent)

        return out_latent


class MotionEncoder2(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        input_indicies: List[int],
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        detach_latent: bool = False,
        prop_layer_norm: bool = False,
        random_input_indicies: int = 0,
    ) -> None:
        super().__init__()

        self.input_adapter = input_adapter
        self.latent_dim = num_latent_channels

        self.input_layer = pio.Sequential(
            pio.cross_attention_layer(
                num_q_channels=num_latent_channels,
                num_kv_channels=input_adapter.num_input_channels,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
            ),
            pio.self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads,
                dropout=dropout,
            ),
        )

        self.update_layer = pio.cross_attention_layer(
            num_q_channels=num_latent_channels,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
        )

        self.propagate_layer = pio.self_attention_block(
            num_layers=num_self_attention_layers_per_block,
            num_channels=num_latent_channels,
            num_heads=num_self_attention_heads,
            dropout=dropout,
        )

        self.input_indicies = set(input_indicies)
        self.random_input_indicies = random_input_indicies

        # detach and clone latent at every output index during
        # training so gradients don't flow all the way back
        self.detach_latent = detach_latent

        self.prop_layer_norm = (
            nn.LayerNorm(num_latent_channels) if prop_layer_norm else None
        )

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        raise NotImplementedError()

    def forward(
        self, output_idx: Tensor, agents: Tensor, agents_mask: Tensor
    ) -> List[Tensor]:
        """Given an input of dims [Time Batch Tokens Channels]"""

        # repeat initial latent vector along batch dimension
        in_latent = einops.repeat(self.latent, "... -> b ...", b=agents.shape[1])

        max_idx = int(output_idx.max().item()) + 1
        first_input_idx = min(self.input_indicies)

        x_adapt = self.input_adapter(
            agents[first_input_idx], agents_mask[first_input_idx]
        )
        x_latent = self.input_layer(in_latent, x_adapt, agents_mask[first_input_idx])

        out_latent = [x_latent] if 0 in output_idx else []

        input_indicies = self.input_indicies
        if self.random_input_indicies > 0 and self.training:
            candidates = list(range(min(input_indicies), max(input_indicies)))
            random.shuffle(candidates)
            input_indicies.update(candidates[: self.random_input_indicies])

        for t_idx in range(first_input_idx + 1, max_idx):
            x_latent = self.propagate_layer(x_latent)
            if t_idx in self.input_indicies:
                x_step, x_mask = agents[t_idx], agents_mask[t_idx]
                x_adapt, x_mask = self.input_adapter(x_step, x_mask)
                update_latent = self.input_layer(in_latent, x_adapt, x_mask)
                x_latent = self.update_layer(x_latent, update_latent)

            if t_idx in output_idx:
                out_latent.append(x_latent)
                if self.detach_latent and self.training:
                    x_latent = x_latent.detach().clone()

            if self.prop_layer_norm is not None:
                x_latent = self.prop_layer_norm(x_latent)

        return out_latent


class MotionEncoder3(MotionEncoder):
    """
    Slight modification from MotionEncoder which always uses propagate
    layer and adds cross-attention update afterward if there's a new measurement
    """

    def __init__(
        self,
        *args,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.update_layer = pio.cross_attention_layer(
            num_q_channels=self.latent_dim,
            num_kv_channels=self.input_adapter.num_input_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
        )

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        raise NotImplementedError()

    def forward(
        self,
        output_idx: Tensor,
        agents: Tensor,
        agents_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """Given an input of dims [Time Batch Tokens Channels]"""

        # repeat initial latent vector along batch dimension
        x_latent: Tensor = einops.repeat(self.latent, "... -> b ...", b=agents.shape[1])

        max_idx = int(output_idx.max().item()) + 1
        min_idx = min(self.input_indicies)
        out_latent = []

        x_adapt, x_mask = self.input_adapter(agents[min_idx], agents_mask[min_idx])
        x_latent = self.input_layer(x_latent, x_adapt, x_mask)

        if min_idx in output_idx:
            out_latent.append(x_latent)
            if self.detach_latent and self.training:
                x_latent = x_latent.detach().clone()

        input_indicies = self.input_indicies
        if self.random_input_indicies > 0 and self.training:
            candidates = list(range(min(input_indicies), max(input_indicies)))
            random.shuffle(candidates)
            input_indicies.update(candidates[: self.random_input_indicies])

        for t_idx in range(min_idx + 1, max_idx):
            x_latent = self.propagate_layer(x_latent)

            if t_idx in input_indicies:
                x_adapt, x_mask = self.input_adapter(agents[t_idx], agents_mask[t_idx])
                x_latent = self.update_layer(x_latent, x_adapt, x_mask)

            if t_idx in output_idx:
                out_latent.append(x_latent)
                if self.detach_latent and self.training:
                    x_latent = x_latent.detach().clone()

            if self.prop_layer_norm is not None:
                x_latent = self.prop_layer_norm(x_latent)

        return out_latent


class MotionEncoder3Ctx(MotionEncoder3):
    r"""Adding context module for roadgraph and traffic
    light features via cross attention after the prediction step"""

    def __init__(
        self,
        *args,
        num_cross_attention_heads: int = 4,
        dropout: float = 0,
        roadgraph_ia: Dict[str, Any] = None,
        signal_ia: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            num_cross_attention_heads=num_cross_attention_heads,
            dropout=dropout,
            **kwargs,
        )

        if signal_ia is not None:
            self.signal_encoder = (
                pio.mlp(signal_ia["channels"])
                if signal_ia["type"] == "mlp"
                else SignalIA(**signal_ia["args"])
            )

            self.signal_attn = pio.Sequential(
                pio.cross_attention_layer(
                    num_q_channels=self.latent_dim,
                    num_kv_channels=self.signal_encoder.num_input_channels,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
                pio.self_attention_layer(
                    num_channels=self.latent_dim,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
            )
        else:
            self.signal_encoder = None
            self.signal_attn = None

        if roadgraph_ia is not None:
            self.roadgraph_encoder = (
                pio.mlp(roadgraph_ia["channels"])
                if roadgraph_ia["type"] == "mlp"
                else ImageIA(**roadgraph_ia["args"])
            )

            self.road_attn = pio.Sequential(
                pio.cross_attention_layer(
                    num_q_channels=self.latent_dim,
                    num_kv_channels=self.roadgraph_encoder.num_input_channels,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
                pio.self_attention_layer(
                    num_channels=self.latent_dim,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
            )
        else:
            self.roadgraph_encoder = None
            self.road_attn = None

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        num_latent = self.latent.shape[0]

        class Adapted(nn.Module):
            def __init__(self, adapter, cross_attn) -> None:
                super().__init__()
                self.adapter = adapter
                self.cross_attn = cross_attn

            def forward(self, latent, data, mask):
                data, mask = self.adapter(data, mask)
                return self.cross_attn(latent, data, mask)

        print("Exporting input_layer")
        inputs = (
            torch.randn((1, num_latent, self.latent_dim)).cuda(),
            torch.ones((1, num_latent, 8)).cuda(),
            torch.ones((1, num_latent)).cuda(),
        )
        input_layer = Adapted(self.input_adapter, self.input_layer)
        torch.onnx.export(input_layer, inputs, path / "input_layer.onnx")

        print("Exporting update_layer")
        inputs = (
            torch.randn((1, num_latent, self.latent_dim)).cuda(),
            torch.ones((1, num_latent, 8)).cuda(),
            torch.ones((1, num_latent)).cuda(),
        )
        update_layer = Adapted(self.input_adapter, self.update_layer)
        torch.onnx.export(update_layer, inputs, path / "update_layer.onnx")

        print("Exporting propagate_layer")
        inputs = torch.randn((1, num_latent, self.latent_dim)).cuda()
        torch.onnx.export(self.propagate_layer, inputs, path / "propagate_layer.onnx")

        if self.roadgraph_encoder is not None:
            print("Exporting road_encoder")
            inputs = torch.randn((1, *self.roadgraph_encoder.image_shape)).cuda()
            torch.onnx.export(
                self.roadgraph_encoder, inputs, path / "road_encoder.onnx"
            )

            print("Exporting road_update")
            num_road_tokens: int = reduce(
                lambda x, y: x * y, self.roadgraph_encoder.spatial_shape
            )
            inputs = (
                torch.randn((1, num_latent, self.latent_dim)).cuda(),
                torch.randn(
                    (1, num_road_tokens, self.roadgraph_encoder.num_input_channels)
                ).cuda(),
                torch.ones(1, num_road_tokens).cuda(),
            )
            torch.onnx.export(self.road_attn, inputs, path / "road_update.onnx")

        if self.signal_encoder is not None:
            print("Exporting signal_update")
            inputs = (
                torch.randn((1, num_latent, self.latent_dim)).cuda(),
                torch.ones((1, 16, 3)).cuda(),
                torch.ones((1, 16)).cuda(),
            )
            signal_update = Adapted(self.signal_encoder, self.signal_attn)
            torch.onnx.export(signal_update, inputs, path / "signal_update.onnx")

        print(f"Finished {type(self).__name__} Export")

    def forward(
        self,
        output_idx: Tensor,
        agents: Tensor,
        agents_mask: Tensor,
        roadgraph: Tensor = None,
        roadgraph_mask: Tensor = None,
        signals: Tensor = None,
        signals_mask: Tensor = None,
    ) -> List[Tensor]:
        """Given an input of dims [Time Batch Tokens Channels]"""

        # repeat initial latent vector along batch dimension
        x_latent: Tensor = einops.repeat(self.latent, "... -> b ...", b=agents.shape[1])

        max_idx = int(output_idx.max().item()) + 1
        min_idx = min(self.input_indicies)
        out_latent = []

        if self.roadgraph_encoder is not None:
            enc_roadgraph = self.roadgraph_encoder(roadgraph)
            if roadgraph_mask is None:
                roadgraph_mask = torch.ones(
                    enc_roadgraph.shape[:-1], device=enc_roadgraph.device
                )

        x_adapt, x_mask = self.input_adapter(agents[min_idx], agents_mask[min_idx])
        x_latent = self.input_layer(x_latent, x_adapt, x_mask)

        if min_idx in output_idx:
            out_latent.append(x_latent)
            if self.detach_latent and self.training:
                x_latent = x_latent.detach().clone()

        input_indicies = self.input_indicies
        if self.random_input_indicies > 0 and self.training:
            candidates = list(range(min(input_indicies), max(input_indicies)))
            random.shuffle(candidates)
            input_indicies.update(candidates[: self.random_input_indicies])

        for t_idx in range(min_idx + 1, max_idx):
            x_latent = self.propagate_layer(x_latent)

            if t_idx in input_indicies:
                x_adapt, x_mask = self.input_adapter(agents[t_idx], agents_mask[t_idx])
                x_latent = self.update_layer(x_latent, x_adapt, x_mask)

                if self.signal_encoder is not None:
                    s_adapt, s_mask = self.signal_encoder(
                        signals[t_idx], signals_mask[t_idx]
                    )
                    x_latent = self.signal_attn(x_latent, s_adapt, s_mask)

            if self.road_attn is not None:
                x_latent = self.road_attn(x_latent, enc_roadgraph, roadgraph_mask)

            if t_idx in output_idx:
                out_latent.append(x_latent)
                if self.detach_latent and self.training:
                    x_latent = x_latent.detach().clone()

            if self.prop_layer_norm is not None:
                x_latent = self.prop_layer_norm(x_latent)

        return out_latent


class _CrossAttnWContext(nn.Module):
    """Allows to inject roadgraph features in the self
    attention after the cross-attention"""

    def __init__(
        self, cross_attn: pio.CrossAttention, self_attn: pio.SelfAttentionStaticFeatures
    ) -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.self_attn = self_attn

    def forward(
        self,
        q,
        kv,
        c,
        in_pad_mask=None,
        in_attn_mask=None,
        ctx_pad_mask=None,
        ctx_attn_mask=None,
    ) -> Tensor:
        """"""
        i = self.cross_attn(q, kv, in_pad_mask, in_attn_mask)
        i = self.self_attn(i, c, ctx_pad_mask, ctx_attn_mask)
        return i


class MotionEncoderContext(nn.Module):
    """
    Adds context from roadgraph features and traffic lights.\n
    If roadgraph input adapter kwargs are specified, then the roadgraph
    context is an image which needs to be tokenised with respective params.
    """

    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        input_indicies: List[int],
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        detach_latent: bool = False,
        prop_layer_norm: bool = False,
        roadgraph_ia: Dict[str, Any] = None,
        signal_ia: Dict[str, Any] = None,
    ) -> None:
        super().__init__()

        self.input_adapter = input_adapter
        self.latent_dim = num_latent_channels

        self.input_layer = _CrossAttnWContext(
            pio.cross_attention_layer(
                num_q_channels=num_latent_channels,
                num_kv_channels=input_adapter.num_input_channels,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
            ),
            pio.self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads,
                dropout=dropout,
                with_context=True,
            ),
        )

        self.propagate_layer = pio.self_attention_block(
            num_layers=num_self_attention_layers_per_block,
            num_channels=num_latent_channels,
            num_heads=num_self_attention_heads,
            dropout=dropout,
            with_context=True,
        )

        self.input_indicies = set(input_indicies)

        # detach and clone latent at every output index during
        # training so gradients don't flow all the way back
        self.detach_latent = detach_latent

        self.prop_layer_norm = (
            nn.LayerNorm(num_latent_channels) if prop_layer_norm else None
        )

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

        self.signal_encoder = (
            pio.mlp(signal_ia["channels"])
            if signal_ia["type"] == "mlp"
            else SignalIA(**signal_ia["args"])
        )
        self.roadgraph_encoder = (
            pio.mlp(roadgraph_ia["channels"])
            if roadgraph_ia["type"] == "mlp"
            else ImageIA(**roadgraph_ia["args"])
        )

    @torch.no_grad()
    def _init_parameters(self):
        self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        raise NotImplementedError()

    def forward(
        self,
        output_idx: Tensor,
        agents: Tensor,
        agents_mask: Tensor,
        roadgraph: Tensor,
        roadgraph_mask: Tensor,
        signals: Tensor,
        signals_mask: Tensor,
    ) -> List[Tensor]:
        """Given an input of dims [Time Batch Tokens Channels]"""

        # repeat initial latent vector along batch dimension
        latent = einops.repeat(self.latent, "... -> b ...", b=agents.shape[1])

        max_idx = int(output_idx.max().item()) + 1
        first_input_idx = min(self.input_indicies)
        out_latent = []

        enc_roadgraph = self.roadgraph_encoder(roadgraph)
        if roadgraph_mask is None:
            roadgraph_mask = torch.ones(
                enc_roadgraph.shape[:-1], device=enc_roadgraph.device
            )

        for t_idx in range(first_input_idx, max_idx):
            if t_idx in self.input_indicies:
                a_step, a_mask = agents[t_idx], agents_mask[t_idx]
                s_step, s_mask = signals[t_idx], signals_mask[t_idx]

                s_step = self.signal_encoder(s_step, s_mask)

                ctx_feats = torch.cat([enc_roadgraph, s_step], dim=1)
                ctx_mask = torch.cat([roadgraph_mask, s_mask], dim=1)

                a_adapt, a_mask = self.input_adapter(a_step, a_mask)
                latent = self.input_layer(
                    latent, a_adapt, ctx_feats, a_mask, None, ctx_mask, None
                )
            else:
                latent = self.propagate_layer(latent, ctx_feats, ctx_mask)

            if t_idx in output_idx:
                out_latent.append(latent)
                if self.detach_latent and self.training:
                    latent = latent.detach().clone()

            if self.prop_layer_norm is not None:
                latent = self.prop_layer_norm(latent)

        return out_latent


class CrossLatentAttention(nn.Module):
    def __init__(
        self,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.ModuleList(
            [
                pio.self_attention_layer(
                    num_latent_channels,
                    num_self_attention_heads,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.cross_attn = nn.ModuleList(
            [
                pio.cross_attention_layer(
                    num_latent_channels,
                    num_latent_channels,
                    num_cross_attention_heads,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, q_latent: Tensor, kv_latent: Tensor) -> Tensor:
        """Perform cross attention between two latent states"""
        for self_attn, cross_attn in zip(self.self_attn, self.cross_attn):
            q_latent = cross_attn(q_latent, kv_latent)
            q_latent = self_attn(q_latent)

        return q_latent


class MotionEncoderParallel(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: Union[List[int], int],
        num_latent_channels: int,
        class_names: List[str],
        input_indicies: List[int],
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 3,
        dropout: float = 0.0,
        detach_latent: bool = False,
    ) -> None:
        super().__init__()

        self.input_adapter = input_adapter
        self.latent_dim = num_latent_channels

        self.initial_layer = nn.ModuleDict()
        self.update_layer = nn.ModuleDict()
        self.propagate_layer = nn.ModuleDict()
        self.cross_class_layer = nn.ModuleDict()

        for name in class_names:
            self.initial_layer[name] = pio.Sequential(
                pio.cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=input_adapter.num_input_channels,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
                pio.self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                ),
            )

            self.update_layer[name] = pio.cross_attention_layer(
                num_q_channels=num_latent_channels,
                num_kv_channels=input_adapter.num_input_channels,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
            )

            self.propagate_layer[name] = pio.self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads,
                dropout=dropout,
            )

            self.cross_class_layer[name] = CrossLatentAttention(
                num_latent_channels=num_latent_channels,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=num_self_attention_heads,
            )

            # Unique x-attn function for each class pair
            # for name_kv in [n for n in class_names if n != name]:
            #     self.cross_class_layer[f"{name}_{name_kv}"] = CrossLatentAttention(
            #         num_latent_channels=num_latent_channels,
            #         num_cross_attention_heads=num_cross_attention_heads,
            #         num_self_attention_heads=num_self_attention_heads,
            #     )

        self.input_indicies = set(input_indicies)

        # detach and clone latent at every output index during
        # training so gradients don't flow all the way back
        self.detach_latent = detach_latent

        if isinstance(num_latents, int):
            num_latents = [num_latents for _ in range(len(class_names))]

        # learnable initial latent vectors
        self.latents = nn.ParameterDict(
            {
                name: torch.empty(n, num_latent_channels)
                for n, name in zip(num_latents, class_names)
            }
        )
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        for latent in self.latents:
            self.latents[latent].normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def onnx_export(self, path: Path) -> None:
        """Export to onnx file"""
        raise NotImplementedError()

    def forward(
        self,
        output_idx: Tensor,
        agents: Tensor,
        agents_mask: Optional[Tensor] = None,
    ):
        """"""
        # repeat initial latent vector along batch dimension
        latents: Dict[str, Tensor] = {
            name: einops.repeat(latent, "... -> b ...", b=agents.shape[1])
            for name, latent in self.latents.items()
        }

        max_idx = int(output_idx.max().item()) + 1
        min_idx = min(self.input_indicies)
        out_latent = []

        x_adapt, x_mask = self.input_adapter(agents[min_idx], agents_mask[min_idx])
        for cls_name in latents:
            latents[cls_name] = self.initial_layer[cls_name](
                latents[cls_name], x_adapt, x_mask[cls_name]
            )

        if min_idx in output_idx:
            out_latent.append(latents)
            if self.detach_latent and self.training:
                latents = {l: latents[l].detach().clone() for l in latents}

        for t_idx in range(min_idx + 1, max_idx):
            for cls_name in latents:
                latents[cls_name] = self.propagate_layer[cls_name](latents[cls_name])

            for cls_name in latents:
                kv_latent = torch.cat(
                    [latents[l] for l in latents if l != cls_name], dim=1
                ).detach()
                latents[cls_name] = self.cross_class_layer[cls_name](
                    latents[cls_name], kv_latent
                )

            if t_idx in self.input_indicies:
                x_adapt, x_mask = self.input_adapter(agents[t_idx], agents_mask[t_idx])
                for cls_name in latents:
                    latents[cls_name] = self.update_layer[cls_name](
                        latents[cls_name], x_adapt, x_mask[cls_name]
                    )

            if t_idx in output_idx:
                out_latent.append(latents)
                if self.detach_latent and self.training:
                    latents = {l: latents[l].detach().clone() for l in latents}

        return out_latent


class MotionPerceiver(nn.Module):
    def __init__(self, encoder: Dict[str, Any], decoder: Dict[str, Any]) -> None:
        super().__init__()

        # Setup Encoder
        enc_version = encoder.pop("version", 1)
        in_adapt = encoder.pop("adapter")
        input_adapter = {"vehicle": TrafficIA}[in_adapt["type"].lower()](
            **in_adapt["args"]
        )
        self.encoder = [
            MotionEncoder,
            MotionEncoder2,
            MotionEncoderContext,
            MotionEncoderParallel,
            MotionEncoder3,
            MotionEncoder3Ctx,
        ][enc_version - 1](input_adapter=input_adapter, **encoder)

        # Setup Decoder
        out_adapt = decoder.pop("adapter")
        out_adapter = {"heatmap": HeatmapOA, "classheatmap": ClassHeatmapOA}[
            out_adapt["type"].lower()
        ](**out_adapt["args"])
        self.decoder = pio.PerceiverDecoder(
            output_adapter=out_adapter,
            num_latent_channels=self.encoder.latent_dim,
            **decoder,
        )

    def onnx_export(self, path: Path = None) -> None:
        """Export to ONNX format"""
        if path is None:
            path = Path.cwd()
            print(f"saving at {path}")

        self.encoder.onnx_export(path)
        self.decoder.onnx_export(path)

    def forward(
        self,
        time_idx: Tensor,
        agents: Tensor,
        agents_valid: Tensor,
        roadgraph: Tensor = None,
        roadgraph_valid: Tensor = None,
        roadmap: Tensor = None,
        signals: Tensor = None,
        signals_valid: Tensor = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Format of x_latent, data and mask is T,B,N,C
        For the time being, the time_idxs to sample from are broadcast
        across the batch.
        """
        kwargs = {
            "output_idx": time_idx[0],
            "agents": agents.moveaxis((2, 0), (0, 1)),
            "agents_mask": ~agents_valid.moveaxis((2, 0), (0, 1)).bool(),
        }

        if roadgraph is not None and roadgraph.size():
            kwargs["roadgraph"] = roadgraph
            kwargs["roadgraph_mask"] = roadgraph_valid.bool()

        if roadmap is not None and roadmap.size():
            kwargs["roadgraph"] = roadmap
            kwargs["roadgraph_mask"] = None

        if signals is not None and signals.size():
            kwargs["signals"] = signals.transpose(0, 1)
            kwargs["signals_mask"] = signals_valid.transpose(0, 1).bool()

        x_latents: List[Tensor] = self.encoder(**kwargs)

        x_logits = []  # [T,{K:[B,H,W]}]
        for x_latent in x_latents:
            x_logits.append(self.decoder(x_latent))

        out_logits = {}  # {K:[B,T,H,W]}
        for logit_name in x_logits[0]:
            out_logits[logit_name] = torch.cat([x[logit_name] for x in x_logits], dim=1)

        return out_logits


def _show_latent(latent, logit, gt, idx):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(latent[idx].cpu())
    plt.savefig(f"latent{idx}.png")
    plt.imshow(logit[idx, 0].cpu())
    plt.savefig(f"pred{idx}.png")
    plt.imshow(gt[idx, 0].cpu())
    plt.savefig(f"truth{idx}.png")
