import math
from typing import Optional, List, Union, Tuple, Sequence, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .common_layer import (
    get_norm_layers,
    get_act_layers,
    GlobalPool
)
from .common_module import (
    InvertedResidual,
    SeparableSelfAttention,
    FFNConv,
)
from . import MODEL_REGISTRIES
from utils.math_utils import make_divisible, bound_fn


class LPViTBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            dropout: Optional[float] = 0.1,
            attn_dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            norm_layer: str = "layer_norm_2d",
            act_layer: str = "hard_swish",
            bias: Optional[bool] = True,
    ) -> None:
        super(LPViTBlock, self).__init__()

        attn_unit1 = SeparableSelfAttention(
            embed_dim=embed_dim,
            attn_dropout=attn_dropout,
            bias=bias
        )

        self.pre_norm_attn1 = nn.Sequential(
            get_norm_layers(num_features=embed_dim, norm_name=norm_layer),
            attn_unit1,
            nn.Dropout(dropout),
        )

        attn_unit2 = SeparableSelfAttention(
            embed_dim=embed_dim,
            attn_dropout=attn_dropout,
            bias=bias
        )

        self.pre_norm_attn2 = nn.Sequential(
            get_norm_layers(num_features=embed_dim, norm_name=norm_layer),
            attn_unit2,
            nn.Dropout(dropout),
        )

        self.pre_norm_ffn = FFNConv(
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            ffn_dropout=ffn_dropout,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x_patch = torch.transpose(x, 2, 3)
        x_patch = self.pre_norm_attn1(x_patch)
        x_patch = torch.transpose(x_patch, 2, 3)
        x = self.pre_norm_attn2(x)
        x = x + x_patch
        x = x + res

        x = x + self.pre_norm_ffn(x)

        return x


class HBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            n_local_blocks: int = 1,
            n_attn_blocks: Optional[int] = 2,
            patch_h: Optional[int] = 8,
            patch_w: Optional[int] = 8,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            attn_dropout: Optional[float] = 0.0,
            norm_layer: Optional[str] = "layer_norm_2d",
            expand_ratio: Optional[Union[int, float, tuple, list]] = 2,
    ) -> None:
        attn_unit_dim = out_channels
        super(HBlock, self).__init__()

        self.local_acq, out_channels = self._build_local_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            n_layers=n_local_blocks,
        )

        self.global_acq, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=norm_layer,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_attn_blocks = n_attn_blocks
        self.n_local_blocks = n_local_blocks

    def _build_local_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Optional[Union[int, float, tuple, list]],
        n_layers: int,
    ) -> Tuple[nn.Module, int]:
        if isinstance(expand_ratio, (int, float)):
            expand_ratio = [expand_ratio] * n_layers
        elif isinstance(expand_ratio, (list, tuple)):
            pass
        else:
            raise NotImplementedError

        local_acq = []
        if stride == 2 and n_layers != 0:
            local_acq.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio[0],
                )
            )

            for i in range(1, n_layers):
                local_acq.append(
                    InvertedResidual(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        expand_ratio=expand_ratio[i],
                    )
                )

        else:
            for i in range(n_layers):
                local_acq.append(
                    InvertedResidual(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                        expand_ratio=expand_ratio[i],
                    )
                )

        return nn.Sequential(*local_acq), out_channels

    def _build_attn_layer(
            self,
            d_model: int,
            ffn_mult: Union[Sequence, int, float],
            n_layers: int,
            attn_dropout: float,
            dropout: float,
            ffn_dropout: float,
            attn_norm_layer: str,
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                    np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_acq = [
            LPViTBlock(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        global_acq.append(
            get_norm_layers(num_features=d_model, norm_name=attn_norm_layer),
        )

        return nn.Sequential(*global_acq), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_acq(x)
        fm_local = fm

        # convert feature map to patches
        patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations on all patches
        patches = self.global_acq(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = fm + fm_local

        return fm


def get_config(scale: Optional[Union[int, float, str, None]] = None) -> dict:
    width_multiplier = scale if scale else 1.0

    ffn_multiplier = (
        2
    )

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(32 * width_multiplier, divisor=16)),  # 128 * 128
            "expand_ratio": 1,
            "stride": 1,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 2,
            "n_attn_blocks": 1,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer2": {
            "out_channels": int(make_divisible(48 * width_multiplier, divisor=16)),  # 64 * 64
            "expand_ratio": [1, 1, 3],
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(80 * width_multiplier, divisor=16)),  # 32 * 32
            "expand_ratio": 3,
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(160 * width_multiplier, divisor=16)),  # 16 * 16
            "expand_ratio": [6, 2.5, 2.3],
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 3,
            "n_attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(320 * width_multiplier, divisor=16)),  # 8 * 8
            "expand_ratio": 6,
            "stride": 2,
            "ffn_multiplier": ffn_multiplier,
            "n_local_blocks": 1,
            "n_attn_blocks": 1,
            "patch_h": 2,
            "patch_w": 2,
        },
        "last_layer_exp_factor": 4,
    }

    return config


@MODEL_REGISTRIES.register(component_name="ehfr_net")
class EHFRNet(nn.Module):
    """
    EHFRNet is a neural network model designed for predicting the binding affinity of peptides to MHC class I molecules.
    It uses a combination of convolutional layers, attention mechanisms, and fully connected layers to process input data.
    """

    def __init__(
            self,
            config,
            num_classes: int = 101,
    ) -> None:
        super(EHFRNet, self).__init__()

        config = get_config(scale=config.EHFR_NET.SCALE)
        image_channels = config["layer0"]["img_channels"]
        out_channels = config["layer0"]["out_channels"]

        self.model_conf_dict = dict()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_hblock(
            input_channel=in_channels, cfg=config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_hblock(
            input_channel=in_channels, cfg=config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_hblock(
            input_channel=in_channels, cfg=config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_hblock(
            input_channel=in_channels, cfg=config["layer4"],
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_hblock(
            input_channel=in_channels, cfg=config["layer5"],
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = nn.Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type='mean', keep_dim=False),
            nn.Linear(in_features=out_channels, out_features=num_classes, bias=True),
        )

    def _make_hblock(
            self, input_channel, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        block = []

        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = 0.0

        block.append(
            HBlock(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=cfg.get("stride", 1),
                ffn_multiplier=ffn_multiplier,
                n_local_blocks=cfg.get("n_local_blocks", 1),
                n_attn_blocks=cfg.get("n_attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=0.0,
                attn_dropout=0.0,
                norm_layer="layer_norm_2d",
                expand_ratio=cfg.get("expand_ratio", 4),
            )
        )

        input_channel = cfg.get("out_channels")

        return nn.Sequential(*block), input_channel

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        x = self.classifier(x)
        return x
