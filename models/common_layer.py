# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ian Zhu
# --------------------------------------------------------


from typing import Optional, List, Union, Tuple

import torch
from torch import Tensor
from torch import nn as nn

from .act_layers import get_act_layers
from .norm_layers import get_norm_layers


def auto_pad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class GlobalPool(nn.Module):
    """
    This layers applies global pooling over a 4D or 5D input tensor

    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
            self,
            pool_type: Optional[str] = "mean",
            keep_dim: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            raise TypeError(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List) -> Tensor:
        if self.pool_type == "rms":  # root mean square
            x = x ** 2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)


class DropPath(nn.Module):
    def __init__(
            self,
            drop_prob: float = 0.,
            **kwargs
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) <  keep_prob

        if keep_prob > 0.0:
            mask = mask.float().div_(keep_prob)
        else:
            mask = torch.zeros_like(mask)

        return x * mask


class GhostConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: int = 1,
            stride: int = 1,
            groups: int = 1,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=auto_pad(kernel_size, None, 1),
                groups=groups
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=5,
                stride=1,
                padding=auto_pad(5, None, 1),
                groups=hidden_channels,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish()
        )

    def forward(self, x) -> Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class SeparableConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = 1,
            dilation: Union[int, Tuple[int, ...]] = 1,
            add_bias: bool = False,
            padding_mode: str = "zeros",
            use_norm: bool = True,
            use_act: bool = True,
            force_map_channels: bool = True
    ) -> None:
        super().__init__()

        self.block = nn.Sequential()

        self.dw_conv = None
        self.pw_conv = None

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=False,
                groups=in_channels,
                padding=auto_pad(kernel_size, None, dilation),
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.block.add_module("dw_conv", self.dw_conv)

        if in_channels != out_channels or force_map_channels:
            self.pw_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    groups=1,
                    bias=add_bias,
                    padding=auto_pad(1, None, 1),
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
                nn.Hardswish() if use_act else nn.Identity(),
            )
            self.block.add_module("pw_conv", self.pw_conv)

    def forward(self, x) -> Tensor:
        x = self.dw_conv(x)
        if self.pw_conv:
            x = self.pw_conv(x)
        return x


class ShuffleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            h_groups: int,
            w_groups: int,
            kernel_size: Optional[Union[int, Tuple[int, int]]] = 1,
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            dilation: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            padding_mode: Optional[str] = "zeros",
            use_norm: Optional[bool] = True,
            norm_name: Optional[Union[str, type, object]] = None,
            use_act: Optional[bool] = True,
            act_name: Optional[Union[str, type, object]] = None,
    ) -> None:
        super().__init__()
        self.h_groups = h_groups
        self.w_groups = w_groups

        if norm_name is None:
            norm_name = "batch_norm_2d"

        if act_name is None:
            act_name ="relu"

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=auto_pad(kernel_size, None, dilation),
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
            ),
            get_norm_layers(num_features=out_channels, norm_name=norm_name) if use_norm else nn.Identity(),
            get_act_layers(act_name=act_name) if use_act else nn.Identity(),
        )


    def forward(self, x: Tensor) -> Tensor:
        y = x
        crop_width = x.size(-1)
        crop_height = x.size(-2)
        if self.h_groups > 1 and self.w_groups > 1:
            h_index = self.shuffle_index(crop_height, self.h_groups)
            w_index = self.shuffle_index(crop_width, self.w_groups)
            y = x[:, :, h_index, :][:, :, :, w_index]
        elif self.h_groups > 1:
            h_index = self.shuffle_index(crop_height, self.h_groups)
            y = x[:, :, h_index, :]
        elif self.w_groups > 1:
            w_index = self.shuffle_index(crop_width, self.w_groups)
            y = x[:, :, :, w_index]

        return self.block(y)

    @staticmethod
    def shuffle_index(len: int, groups: int) -> Tensor:
        if len // groups < 2:
            return torch.tensor([i for i in range(len)])
        elif len % groups == 0:
            return torch.tensor([i for i in range(len)]).view(groups, -1).t().flatten(0, -1)
        elif len % groups % 2 == 0:
            return torch.hstack([
                torch.tensor([i for i in range(len % groups // 2)]),
                torch.tensor([i for i in range(len % groups // 2, len - len % groups // 2)]).view(groups,
                                                                                                  -1).t().flatten(0,
                                                                                                                  -1),
                torch.tensor([i for i in range(len - len % groups // 2, len)])
            ])
        else:
            return torch.hstack([
                torch.tensor([i for i in range(len % groups // 2)]),
                torch.tensor([i for i in range(len % groups // 2, len - len % groups // 2 - 1)]).view(groups,
                                                                                                      -1).t().flatten(0,
                                                                                                                      -1),
                torch.tensor([i for i in range(len - len % groups // 2 - 1, len)])
            ])


class ShuffleSeparableConv(SeparableConv):
    def __init__(
        self,
        h_groups: int,
        w_groups: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = 3,
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        add_bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        force_map_channels: Optional[bool] = True
    ) -> None:
        self.h_groups = h_groups
        self.w_groups = w_groups

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            add_bias=add_bias,
            padding_mode=padding_mode,
            use_norm=use_norm,
            use_act=use_act,
            force_map_channels=force_map_channels,
        )

        self.dw_conv = ShuffleConv(
            in_channels=in_channels,
            out_channels=in_channels,
            h_groups=h_groups,
            w_groups=w_groups,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            padding_mode=padding_mode,
            use_norm=True,
            use_act=False,
        )


class PartialConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            divide_num: int,
    ) -> None:
        super().__init__()
        self.hidden_channels = in_channels // divide_num
        self.untorched_channels = in_channels - self.hidden_channels
        self.partial_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=auto_pad(3, None, 1),
            bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.hidden_channels, self.untorched_channels], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), dim=1)

        return x
