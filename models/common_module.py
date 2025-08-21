from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .common_layer import (
    get_norm_layer,
    get_act_layer,
    auto_pad,
    GhostConv2d,
    ShuffleConv,
)
from utils.math_utils import make_divisible, bound_fn


class GhostBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: Union[int, float],
        dw_kernel_size: int = 3,
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        act_fn_name: Optional[str] = "hard_swish",
        kernel_size: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        act_fn = nn.Hardswish()

        super().__init__()

        # Calculate the middle_channels dynamically
        middle_channels = int(round(in_channels * ratio))

        block = nn.Sequential()

        block.add_module(
            name="ghost1",
            module=GhostConv2d(
                in_channels=in_channels,
                out_channels=middle_channels,
                stride=1,
                kernel_size=kernel_size,
            ),
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if stride > 1:
            conv_dw_bn =nn.Sequential(
                nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=middle_channels,
                    kernel_size=dw_kernel_size,
                    stride=stride,
                    groups=middle_channels,
                    bias=False,
                ),
                get_norm_layer(num_features=middle_channels, norm_type="batch_norm_2d"),
            )
            block.add_module(
                name="conv_dw_bn",
                module=conv_dw_bn
            )

        block.add_module(
            name="ghost2",
            module=nn.Conv2d(
                in_channels=middle_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
            ),
        )

        if (in_channels == out_channels) and (stride == 1):
            shortcut = nn.Identity()
        else:
            shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=dw_kernel_size,
                    stride=stride,  # stride
                    groups=in_channels,  # inchannels
                    bias=False,
                ),
                get_norm_layer(num_features=in_channels, norm_type="batch_norm_2d"),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1, # 1
                    bias=False,
                ),
                get_norm_layer(num_features=out_channels, norm_type="batch_norm_2d"),
            )

        self.block = block
        self.shortcut = shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw_kernel_size = dw_kernel_size
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.act_fn_name = act_fn_name
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y1 = self.block(x)
        y2 = self.shortcut(x)
        return y1 + y2

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, dw_kernel_size={}, stride={}, dilation={}, use_se={}, kernel_size={}, act_fn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.dw_kernel_size,
            self.stride,
            self.dilation,
            self.use_se,
            self.kernel_size,
            self.act_fn_name,
        )


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: Union[int, float] = 2,
            dilation: int = 1,
            skip_connection: Optional[bool] = True,
            act_layer: str = "hard_swish",
    ) -> None:
        assert stride in [1, 2], "The stride should be 1 or 2 in the inverted residual block."
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super(InvertedResidual, self).__init__()

        block = []
        if expand_ratio != 1:
            block.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    get_act_layer(act_name=act_layer),
                )
            )

        block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    padding=auto_pad(3, None, dilation),
                    bias=False,
                    groups=hidden_dim,
                ),
                nn.BatchNorm2d(hidden_dim),
                get_act_layer(act_name=act_layer),
            )
        )

        block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.block = nn.Sequential(*block)
        self.use_res_connect = (
                stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class ShuffleInvertedResidual(InvertedResidual):
    def __init__(
            self,
            h_groups: int,
            w_groups: int,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],
            dilation: int = 1,
            skip_connection: Optional[bool] = True
    ) -> None:
        self.h_groups = h_groups
        self.w_groups = w_groups

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            dilation=dilation,
            skip_connection=skip_connection
        )

        tar_idx = 0 if expand_ratio == 1 else 1

        self.block[tar_idx] = ShuffleConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            h_groups=h_groups,
            w_groups=w_groups,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            groups=self.hidden_dim,
            use_act=True,
            use_norm=True
        )


class SeparableSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
    ):
        super(SeparableSelfAttention, self).__init__()
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + 2 * embed_dim,
            kernel_size=1,
            bias=bias
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=bias
        )

        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_self_attn(x)


class FFN(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            ffn_dropout: Optional[float] = 0.0,
            act_layer: str = "hard_swish",
            norm_layer: str = "layer_norm_2d",
    ) -> None:
        super(FFN, self).__init__()

        self.pre_norm_ffn = nn.Sequential(
            get_norm_layer(num_features=embed_dim, norm_type=norm_layer),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True
            ),
            get_act_layer(act_name=act_layer),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x