from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn

from .common_layer import (
    auto_pad,
    DropPath,
    get_act_layers,
    get_norm_layers,
    PartialConv,
)


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            n_div: int,
            mlp_ratio: float,
            drop_path: float,
            layer_scale_init_value: Optional[Union[int, float]],
            act_layer: str,
            norm_layer: str,
            pconv_fw_type: str
    ) -> None:

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(
                in_channels=dim, 
                out_channels=mlp_hidden_dim, 
                kernel_size=1,
                padding=auto_pad(1, None, 1),
                bias=False
            ),
            get_norm_layers(num_features=mlp_hidden_dim, norm_name=norm_layer),
            get_act_layers(act_name=act_layer),
            nn.Conv2d(
                in_channels=mlp_hidden_dim,
                out_channels=dim,
                kernel_size=1,
                padding=auto_pad(1, None, 1),
                bias=False
            )
        ]
        # conv+BN+ReLu+Conv
        self.mlp = nn.Sequential(*mlp_layer)
        # PConv
        # TODO: pconv_fw_type is not defined.
        self.spatial_mixing = PartialConv(
            in_channels=dim,
            divide_num=n_div,
            # pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(
            self,
            dim: int,
            depth: int,
            n_div: int,
            mlp_ratio: float,
            drop_path: List[float],
            layer_scale_init_value: Optional[Union[int, float]],
            norm_layer: str,
            act_layer: str,
            pconv_fw_type: str
    ) -> None:
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


def get_config(model_scale: str = 's1'):
    config = {
        's1': {
            'input_channels': 3,
            'Stage1': {
                'dim': 64,
                'depth': 2,
                'n_div': 1,
                'mlp_ratio': 4.0,
                'layer_scale_init_value': 1e-5
            }
        }
    }
