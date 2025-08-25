import math
from typing import Sequence, Optional, Tuple, Union, List

import torch
from torch import Tensor, nn

from .common_layer import (
    auto_pad,
    get_norm_layers,
    GlobalPool,
    SeparableConv,
    ShuffleConv,
    ShuffleSeparableConv
)
from .common_module import (
    InvertedResidual,
    ShuffleInvertedResidual
)
from . import MODEL_REGISTRIES
from utils.math_utils import make_divisible, bound_fn


class AggregateLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            dim: Union[int, Tuple[int, int]],  # -1 -2 [-1, -2]
            use_linear: Optional[bool] = True,
            add_bias: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.linear_layer = None
        if use_linear:
            if isinstance(dim, Sequence):
                self.linear_layer = nn.Linear(in_features=in_channels, out_features=in_channels, bias=add_bias)
            else:
                self.linear_layer = nn.Conv1d(
                    in_channels=in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, bias=add_bias, groups=in_channels,
                    padding=auto_pad(1, None, 1),
                )

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == -1 or self.dim == -2:
            y = torch.mean(x, dim=self.dim)
        else:
            y = torch.mean(x, (-2, -1))

        if self.linear_layer is not None:
            y = self.linear_layer(y)

        if self.dim == -1 or self.dim == -2:
            y = torch.unsqueeze(y, self.dim)
        else:
            y.unsqueeze(-1).unsqueeze(-1)

        return y


class AggregateBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            agt_type: Optional[str] = "both",  # channel "height" (dim=-1) "width" (dim=-2) "both"
            use_linear: Optional[bool] = True,
            add_bias: Optional[bool] = False,
            use_norm: Optional[bool] = True,
            use_act: Optional[bool] = False,
    ):
        super().__init__()

        self.agt_channel = None
        self.agt_height = None
        self.agt_width = None
        self.agt_type = agt_type

        if agt_type == "channel":
            self.agt_channel = AggregateLayer(
                in_channels=in_channels,
                dim=(-2, -1),
                use_linear=use_linear,
                add_bias=add_bias,
            )
        elif agt_type == "both":
            self.agt_height = AggregateLayer(
                in_channels=in_channels,
                dim=-1,
                use_linear=use_linear,
                add_bias=add_bias,
            )
            self.agt_width = AggregateLayer(
                in_channels=in_channels,
                dim=-2,
                use_linear=use_linear,
                add_bias=add_bias,
            )
        elif agt_type == "height":
            self.agt_height = AggregateLayer(
                in_channels=in_channels,
                dim=-1,
                use_linear=use_linear,
                add_bias=add_bias,
            )
        elif agt_type == "width":
            self.agt_width = AggregateLayer(
                in_channels=in_channels,
                dim=-2,
                use_linear=use_linear,
                add_bias=add_bias,
            )

        self.norm_layer = get_norm_layers(num_features=in_channels, norm_name="batch_norm_2d") \
            if use_norm else None

        self.act_layer = nn.Hardswish() if use_act else None

    def forward(self, x: Tensor) -> Tensor:
        y = None
        if self.agt_type == "channel":
            y = x + self.agt_channel(x)
        elif self.agt_type == "both":
            y = x + self.agt_height(x) + self.agt_width(x)
        elif self.agt_type == "height":
            y = x + self.agt_height(x)
        elif self.agt_type == "width":
            y = x + self.agt_width(x)

        if self.norm_layer is not None:
            y = self.norm_layer(y)
        if self.act_layer is not None:
            y = self.act_layer(y)

        return y


class LocalGlobalAggregate(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = cfg.get("out_channels")
        self.kernel_size = cfg.get("kernel_size", 3)
        self.stride = cfg.get("stride", 1)
        self.dilation = cfg.get("dilation", 1)
        self.groups = cfg.get("groups", 1)

        self.loc = None
        self.gbl = None
        self.agt = None

        if "loc" in cfg:
            loc_cfg = cfg["loc"]
            block = loc_cfg["block"]
            expand_ratio = loc_cfg.get("expand_ratio")
            use_hs = loc_cfg.get("use_hs")
            use_se = loc_cfg.get("use_se")
            self.loc = self._make_loc_block(block, expand_ratio, use_hs, use_se)
        if "gbl" in cfg:
            gbl_cfg = cfg["gbl"]
            block = gbl_cfg["block"]
            expand_ratio = gbl_cfg.get("expand_ratio")
            use_hs = gbl_cfg.get("use_hs")
            use_se = gbl_cfg.get("use_se")
            h_groups = gbl_cfg.get("h_groups")
            w_groups = gbl_cfg.get("w_groups")
            self.gbl = self._make_gbl_block(h_groups, w_groups, block, expand_ratio, use_hs, use_se)
        if "agt" in cfg:
            agt_cfg = cfg["agt"]
            agt_type = agt_cfg["type"]
            use_linear = agt_cfg["use_linear"]
            add_bias = agt_cfg["add_bias"]
            use_norm = agt_cfg["use_norm"]
            use_act = agt_cfg["use_act"]
            self.agt = AggregateBlock(
                in_channels=in_channels,
                agt_type=agt_type,
                use_linear=use_linear,
                add_bias=add_bias,
                use_norm=use_norm,
                use_act=use_act
            )

    def _make_loc_block(
            self,
            block,
            expand_ratio,
            use_hs,
            use_se,
    ):
        if block == "v0":
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    padding=auto_pad(self.kernel_size, None, self.dilation),
                ),
                nn.BatchNorm2d(num_features=self.out_channels),
                nn.Hardswish()
            )
        if block == "v1":
            return SeparableConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                add_bias=False,
                use_norm=True,
                use_act=True,
                force_map_channels=True
            )
        if block == "v2":
            return InvertedResidual(
                expand_ratio=expand_ratio,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=self.stride,
                dilation=self.dilation,
                skip_connection=True,
            )
        return nn.Identity()
        # if block == "v3":
        #     return InvertedResidualSE(
        #         expand_ratio=expand_ratio,
        #         use_hs=use_hs,
        #         use_se=use_se,
        #         in_channels=self.in_channels,
        #         out_channels=self.out_channels,
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         dilation=self.dilation,
        #     )

    def _make_gbl_block(
            self,
            h_groups,
            w_groups,
            block,
            expand_ratio,
            use_hs,
            use_se,
    ):
        if block == "v0":
            return ShuffleConv(
                h_groups=h_groups,
                w_groups=w_groups,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                use_norm=True,
                use_act=True,
            )
        if block == "v1":
            return ShuffleSeparableConv(
                h_groups=h_groups,
                w_groups=w_groups,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                add_bias=False,
                use_norm=True,
                use_act=True,
                force_map_channels=True
            )
        if block == "v2":
            return ShuffleInvertedResidual(
                h_groups=h_groups,
                w_groups=w_groups,
                expand_ratio=expand_ratio,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=self.stride,
                dilation=self.dilation,
                skip_connection=True,
            )
        return nn.Identity()
        # if block == "v3":
        #     return ShuffleInvertedResidualSE(
        #         h_groups=h_groups,
        #         w_groups=w_groups,
        #         expand_ratio=expand_ratio,
        #         use_hs=use_hs,
        #         use_se=use_se,
        #         in_channels=self.in_channels,
        #         out_channels=self.out_channels,
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         dilation=self.dilation,
        #     )

    def forward(self, x: Tensor) -> Tensor:
        if self.loc is not None and self.gbl is not None and self.agt is not None:
            return self.loc(x) + self.gbl(self.agt(x))

        if self.loc is not None and self.gbl is not None:
            return self.loc(x) + self.gbl(x)

        if self.loc is not None and self.agt is not None:
            return self.loc(self.agt(x))

        if self.gbl is not None and self.agt is not None:
            return self.gbl(self.agt(x))

        if self.loc is not None:
            return self.loc(x)

        if self.gbl is not None:
            return self.gbl(x)

        if self.agt is not None:
            return self.agt(x)


def get_configuration():
    config = {
        # an example
        "example_layer": [
            {
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
                "loc0": {"block": "v0", "groups": 1},
                "loc1": {"block": "v1", "force_map_channels": True},
                "loc2": {"block": "v2", "expand_ratio": 1.0},
                'gbl': {"block": "v3", "expand_ratio": 1.0, "use_hs": False, "use_se": False, "h_groups": 16,
                        "w_groups": 16},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True,
                        "norm_layer": "BatchNorm2d", "use_act": False, "act_name": "SoftReLU"},
            },
            ...
        ],

        "image_channels": 3,
        "head_out_channels": 16,
        "last_channels": 1280,

        # 128x128
        "layer1": [
            {
                "kernel_size": 3,
                "out_channels": 16,
                "stride": 1,
                "loc": {"block": "v2", "expand_ratio": 1},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
        ],

        # 128x128
        "layer2": [
            {
                "kernel_size": 3,
                "out_channels": 24,
                "stride": 2,
                "loc": {"block": "v2", "expand_ratio": 4},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
            # 64x64
            {
                "kernel_size": 3,
                "out_channels": 24,
                "stride": 1,
                "loc": {"block": "v2", "expand_ratio": 3},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
        ],

        # 64x64
        "layer3": [
            {
                "kernel_size": 3,
                "out_channels": 40,
                "stride": 2,
                "loc": {"block": "v2", "expand_ratio": 3},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
            # 32x32
            {
                "kernel_size": 3,
                "out_channels": 40,
                "stride": 1,
                "loc": {"block": "v2", "expand_ratio": 3},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
            # 32x32
            {
                "kernel_size": 3,
                "out_channels": 40,
                "stride": 1,
                "loc": {"block": "v2", "expand_ratio": 3},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
        ],

        # 32x32
        "layer4": [
            {
                "kernel_size": 3,
                "out_channels": 80,
                "stride": 2,
                "loc": {"block": "v2", "expand_ratio": 6},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
            # 16x16
            {
                "kernel_size": 3,
                "out_channels": 80,
                "stride": 1,
                "loc": {"block": "v2", "expand_ratio": 6},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
        ],

        # 16x16
        "layer5": [
            {
                "kernel_size": 3,
                "out_channels": 160,
                "stride": 2,
                "loc": {"block": "v2", "expand_ratio": 6},
                'agt': {"type": "both", "use_linear": True, "add_bias": False, "use_norm": True, "use_act": False},
            },
        ],
    }
    return config


@MODEL_REGISTRIES.register(component_name="af_net")
class MobileAggregate(nn.Module):
    def __init__(
            self,
            config,
            num_classes: int = 101,
    ) -> None:
        super().__init__()

        self.layer_list = None
        self.head = None
        self.body = None
        self.tail = None
        self.classifier = None
        self.round_nearest = 8

        self.width_multiplier = config.AF_NET.SCALE
        self.n_classes = num_classes
        self.classifier_dropout = 0.0

        self.model_conf_dict = dict()

        if self.classifier_dropout == 0.0 or self.classifier_dropout is None:
            val = round(0.2 * self.width_multiplier, 3)
            self.classifier_dropout = bound_fn(min_val=0.0, max_val=0.2, value=val)

        # get net structure configuration
        self.cfg = get_configuration()

        # create configuration dict
        self.model_conf_dict: dict = dict()

        output_stride = None
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        # build net
        self._make_head_layer()
        self._make_body_layers()
        self._make_tail_layers()
        self._make_classifier_layer()

    def _make_head_layer(self):
        image_channels = self.cfg.get("image_channels", 3)
        self.cur_channels = self.cfg.get("head_out_channels", 16)

        self.head = nn.Sequential(
            AggregateBlock(
                in_channels=image_channels,
                agt_type="both",
            ),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.cur_channels,
                kernel_size=3,
                stride=2,
                padding=auto_pad(3, None, 1),
            ),
            nn.BatchNorm2d(self.cur_channels),
            nn.Hardswish()
        )

        self.model_conf_dict["head"] = {"in": image_channels, "out": self.cur_channels}

    def _make_body_layers(self):
        self.body = nn.Sequential()
        layer_1, out_channels = self._make_layer(
            configure=self.cfg["layer1"],
            input_channel=self.cur_channels,
        )
        self.model_conf_dict["layer1"] = {"in": self.cur_channels, "out": out_channels}
        self.cur_channels = out_channels
        self.body.add_module(name="layer1", module=layer_1)

        layer_2, out_channels = self._make_layer(
            configure=self.cfg["layer2"],
            input_channel=self.cur_channels,
        )
        self.model_conf_dict["layer2"] = {"in": self.cur_channels, "out": out_channels}
        self.cur_channels = out_channels
        self.body.add_module(name="layer2", module=layer_2)

        layer_3, out_channels = self._make_layer(
            configure=self.cfg["layer3"],
            input_channel=self.cur_channels,
        )
        self.model_conf_dict["layer3"] = {"in": self.cur_channels, "out": out_channels}
        self.cur_channels = out_channels
        self.body.add_module(name="layer3", module=layer_3)

        layer_4, out_channels = self._make_layer(
            configure=self.cfg["layer4"],
            input_channel=self.cur_channels,
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": self.cur_channels, "out": out_channels}
        self.cur_channels = out_channels
        self.body.add_module(name="layer4", module=layer_4)

        layer_5, out_channels = self._make_layer(
            configure=self.cfg["layer5"],
            input_channel=self.cur_channels,
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": self.cur_channels, "out": out_channels}
        self.cur_channels = out_channels
        self.body.add_module(name="layer5", module=layer_5)

    def _make_tail_layers(self):
        last_channels = self.cfg.get("last_channels", 1024)
        last_channels = make_divisible(
            last_channels * max(1.0, self.width_multiplier), self.round_nearest
        )

        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=self.cur_channels,
                out_channels=last_channels,
                kernel_size=1,
                stride=1,
                padding=auto_pad(1, None, 1),
            ),
            nn.BatchNorm2d(last_channels),
            nn.Hardswish()
        )

        self.model_conf_dict["tail"] = {"in": self.cur_channels, "out": last_channels}

        self.cur_channels = last_channels

    def _make_classifier_layer(self):

        self.classifier = nn.Sequential(
            GlobalPool(pool_type='mean', keep_dim=False),
            nn.Linear(in_features=self.cur_channels, out_features=self.n_classes, bias=True),
        )

        self.model_conf_dict["cls"] = {"in": self.cur_channels, "out": self.n_classes}
        self.cur_channels = self.n_classes

    def _make_layer(
            self,
            configure,
            input_channel: int,
            dilate: Optional[bool] = False,
    ) -> Tuple[nn.Module, int]:
        # prev_dilation = self.dilation
        block_seq = nn.Sequential()
        count = 0

        for cfg_block in configure:
            out_channels_cfg = cfg_block.get("out_channels")
            stride = cfg_block.get("stride", 1)
            cfg_block["out_channels"] = make_divisible(out_channels_cfg * self.width_multiplier, self.round_nearest)
            if dilate and count == 0:
                self.dilation *= stride
                stride = 1
                cfg_block["stride"] = 1

            block = LocalGlobalAggregate(
                in_channels=input_channel,
                cfg=cfg_block,
            )
            block_seq.add_module(
                name=f"ms_s_{stride}_idx-{count}",
                module=block
            )
            count += 1
            input_channel = cfg_block["out_channels"]

        return block_seq, input_channel

    def forward(self, x):
        if self.layer_list:
            self.layer_list = self.layer_list if isinstance(self.layer_list, nn.Sequential) else nn.Sequential(
                *self.layer_list)
            x = self.layer_list(x)
        else:
            if self.head:
                x = self.head(x)
            if self.body:
                self.body = self.body if isinstance(self.body, nn.Sequential) else nn.Sequential(*self.body)
                x = self.body(x)
            if self.tail:
                x = self.tail(x)
            if self.classifier:
                x = self.classifier(x)

        return x
