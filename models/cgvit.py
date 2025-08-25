# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ian Zhu
# --------------------------------------------------------

from typing import List, Optional, Union

import warnings

import torch
from torch import Tensor
import torch.nn as nn

from .common_layer import (
    auto_pad,
    DropPath,
    get_act_layers,
    get_norm_layers,
    GlobalPool,
    PartialConv,
)

from .common_module import (
    GhostBottleneck,
    PartialBlock,
)

from .mobilevit import MobileViTBlock
from .mobilevitv2 import MobileViTBlockV2

from . import MODEL_REGISTRIES


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


def get_config(scale: str = 's1'):
    config = {
        's1': {
            'input_channels': 3,
            'output_channels': 960,
            'Stage1': {
                'dim': 32,
                'kernel_size': 1,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 2,
                'vit': None,
            },
            'Stage2': {
                'dim': 64,
                'kernel_size': 1,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 4,
                'vit': None,
            },
            'Stage3': {
                'dim': 96,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 6,
                'vit': {
                    'type': 'v2',
                    'dim': 96,
                    'patch_h': 2,
                    'patch_w': 2,
                    'depth': 2,
                    'drop_path_rate': 0.2,
                },
            },
            'Stage4': {
                'dim': 128,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 6,
                'vit': {
                    'type': 'v2',
                    'dim': 128,
                    'patch_h': 2,
                    'patch_w': 2,
                    'depth': 2,
                    'drop_path_rate': 0.2,
                },
            },
            'Stage5': {
                'dim': 256,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 6,
                'vit': {
                    'type': 'v2',
                    'dim': 256,
                    'patch_h': 2,
                    'patch_w': 2,
                    'depth': 2,
                    'drop_path_rate': 0.2,
                },
            },
        },
        's2': {
            'input_channels': 3,
            'output_channels': 960,
            'Stage1': {
                'dim': 32,
                'kernel_size': 1,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 2,
                'vit': None,
            },
            'Stage2': {
                'dim': 64,
                'kernel_size': 1,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 4,
                'vit': None,
            },
            'Stage3': {
                'dim': 96,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 5,
                'vit': None,
            },
            'Stage4': {
                'dim': 128,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 6,
                'vit': None,
            },
            'Stage5': {
                'dim': 256,
                'kernel_size': 3,
                'stride': 2,
                'act_layer': 'relu',
                'depth': 5,
                'vit': {
                    'type': 'v1',
                    'dim': 256,
                    'patch_h': 2,
                    'patch_w': 2,
                    'depth': 2,
                    'drop_path_rate': 0.2,
                },
            },
        },
    }

    return config[scale]


@MODEL_REGISTRIES.register(component_name="cgvit")
class CGViT(nn.Module):
    def __init__(
            self,
            config,
            num_classes: int = 101,
    ) -> None:
        super().__init__()

        config = get_config(scale=config.CGVIT.SCALE)
        self.num_classes = num_classes

        stage1_cfg = config['Stage1']
        input_channels = config['input_channels']
        self.stage1 = nn.Sequential(
            GhostBottleneck(
                in_channels=input_channels,
                out_channels=stage1_cfg['dim'],
                kernel_size=stage1_cfg['kernel_size'],
                stride=stage1_cfg['stride'],
                act_fn_name=stage1_cfg['act_layer'],
            ),
            *[
                PartialBlock(
                    in_channels=stage1_cfg['dim'],
                )
                for _ in range(stage1_cfg['depth'])
            ],
            self.build_attn_layers(stage1_cfg['vit']) if stage1_cfg['vit'] else nn.Identity()
        )

        stage2_cfg = config['Stage2']
        input_channels = stage1_cfg['dim']
        self.stage2 = nn.Sequential(
            GhostBottleneck(
                in_channels=input_channels,
                out_channels=stage2_cfg['dim'],
                kernel_size=stage2_cfg['kernel_size'],
                stride=stage2_cfg['stride'],
                act_fn_name=stage2_cfg['act_layer'],
            ),
            *[
                PartialBlock(
                    in_channels=stage2_cfg['dim'],
                )
                for _ in range(stage2_cfg['depth'])
            ],
            self.build_attn_layers(stage2_cfg['vit']) if stage2_cfg['vit'] else nn.Identity()
        )

        stage3_cfg = config['Stage3']
        input_channels = stage2_cfg['dim']
        self.stage3 = nn.Sequential(
            GhostBottleneck(
                in_channels=input_channels,
                out_channels=stage3_cfg['dim'],
                kernel_size=stage3_cfg['kernel_size'],
                stride=stage3_cfg['stride'],
                act_fn_name=stage3_cfg['act_layer'],
            ),
            *[
                PartialBlock(
                    in_channels=stage3_cfg['dim'],
                )
                for _ in range(stage3_cfg['depth'])
            ],
            self.build_attn_layers(stage3_cfg['vit']) if stage3_cfg['vit'] else nn.Identity()
        )

        stage4_cfg = config['Stage4']
        input_channels = stage3_cfg['dim']
        self.stage4 = nn.Sequential(
            GhostBottleneck(
                in_channels=input_channels,
                out_channels=stage4_cfg['dim'],
                kernel_size=stage4_cfg['kernel_size'],
                stride=stage4_cfg['stride'],
                act_fn_name=stage4_cfg['act_layer'],
            ),
            *[
                PartialBlock(
                    in_channels=stage4_cfg['dim'],
                )
                for _ in range(stage4_cfg['depth'])
            ],
            self.build_attn_layers(stage4_cfg['vit']) if stage4_cfg['vit'] else nn.Identity()
        )

        stage5_cfg = config['Stage5']
        input_channels = stage4_cfg['dim']
        self.stage5 = nn.Sequential(
            GhostBottleneck(
                in_channels=input_channels,
                out_channels=stage5_cfg['dim'],
                kernel_size=stage5_cfg['kernel_size'],
                stride=stage5_cfg['stride'],
                act_fn_name=stage5_cfg['act_layer'],
            ),
            *[
                PartialBlock(
                    in_channels=stage5_cfg['dim'],
                )
                for _ in range(stage5_cfg['depth'])
            ],
            self.build_attn_layers(stage5_cfg['vit']) if stage5_cfg['vit'] else nn.Identity()
        )

        out_channels = config['output_channels']

        self.conv_1x1_exp = nn.Sequential(
            nn.Conv2d(
                in_channels=stage5_cfg['dim'],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=auto_pad(1, None, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            GlobalPool(pool_type='mean', keep_dim=False),
            nn.Linear(in_features=out_channels, out_features=num_classes, bias=True),
        )
    
    @staticmethod
    def build_attn_layers(config) -> nn.Module:
        if config['type'] == 'v1':
            attn_block = MobileViTBlock(
                in_channels=config['dim'],
                transformer_dim=config['dim'],
                ffn_dim=config['dim'],
                n_transformer_blocks=config['depth'],
                patch_h=config['patch_h'],
                patch_w=config['patch_w'],
                dropout=config['drop_path_rate'],
            )
        elif config['type'] == 'v2':
            attn_block = MobileViTBlockV2(
                in_channels=config['dim'],
                attn_unit_dim=config['dim'],
                patch_h=config['patch_h'],
                patch_w=config['patch_w'],
            )
        else:
            warnings.warn(f"Unknown attention block type: {config['type']}, we will use Identity instead.")
            attn_block = nn.Identity()

        return attn_block

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        y = self.conv_1x1_exp(x)
        out = self.classifier(y)

        return out
        
