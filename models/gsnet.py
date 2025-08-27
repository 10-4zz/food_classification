from typing import Optional, Tuple, Dict

import torch.nn as nn

from .common_layer import (
	auto_pad,
	get_act_layers,
	get_norm_layers,
    GlobalPool,
)

from .afnet import (
	LocalGlobalAggregate
)

from . import MODEL_REGISTRIES

from utils.math_utils import make_divisible, bound_fn


def get_configuration() -> Dict:

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
				'gbl': {"block": "v3", "expand_ratio": 1.0, "use_hs": False, "use_se": False, "h_groups": 16, "w_groups": 16},
				'agt': {"type": "channel", "size": (3, 128, 128), "use_linear": False,
						"bias": False, "use_norm": True, "use_act": True, "act_name": "relu",
						"use_dropout": False, "dropout": 0.0},
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
				"gbl": {"block": "v2", "expand_ratio": 1, "h_groups": 16, "w_groups": 16},
			},
		],

		# 128x128
		"layer2": [
			{
				"kernel_size": 3,
				"out_channels": 24,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 4},
				"gbl": {"block": "v2", "expand_ratio": 4, "h_groups": 16, "w_groups": 16},
			},
			# 64x64
			{
				"kernel_size": 3,
				"out_channels": 24,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 8, "w_groups": 8},
			},
		],

		# 64x64
		"layer3": [
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 8, "w_groups": 8},
			},
			# 32x32
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 4, "w_groups": 4},
			},
			# 32x32
			{
				"kernel_size": 3,
				"out_channels": 40,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 3},
				"gbl": {"block": "v2", "expand_ratio": 3, "h_groups": 4, "w_groups": 4},
			},
		],

		# 32x32
		"layer4": [
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 6},
				"gbl": {"block": "v2", "expand_ratio": 6, "h_groups": 4, "w_groups": 4},
			},
			# 16x16
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 2.5},
				"gbl": {"block": "v2", "expand_ratio": 2.5, "h_groups": 4, "w_groups": 4},
			},
			# 16x16
			{
				"kernel_size": 3,
				"out_channels": 80,
				"stride": 1,
				"loc": {"block": "v2", "expand_ratio": 2.3},
				"gbl": {"block": "v2", "expand_ratio": 2.3, "h_groups": 4, "w_groups": 4},
			},

		],

		# 16x16
		"layer5": [
			{
				"kernel_size": 3,
				"out_channels": 160,
				"stride": 2,
				"loc": {"block": "v2", "expand_ratio": 6},
				"gbl": {"block": "v2", "expand_ratio": 6, "h_groups": 4, "w_groups": 4},
			},

		],
	}
	return config


@MODEL_REGISTRIES.register(component_name="gs_net")
class MobileShuffleV2(nn.Module):
	def __init__(
			self,
			config,
			num_classes: int = 101,
	) -> None:
		super().__init__()

		self.width_multiplier = config.GS_NET.SCALE
		self.n_classes = num_classes
		self.classifier_dropout = 0.0
		self.round_nearest = 8

		output_stride = None
		self.dilate_l4 = False
		self.dilate_l5 = False
		if output_stride == 8:
			self.dilate_l4 = True
			self.dilate_l5 = True
		elif output_stride == 16:
			self.dilate_l5 = True

		if self.classifier_dropout == 0.0 or self.classifier_dropout is None:
			val = round(0.2 * self.width_multiplier, 3)
			self.classifier_dropout = bound_fn(min_val=0.0, max_val=0.2, value=val)

		# get net structure configuration
		self.cfg = get_configuration()

		# create configuration dict
		self.model_conf_dict: dict = dict()

		# build net
		self._make_head_layer()
		self._make_body_layers()
		self._make_tail_layers()
		self._make_classifier_layer()

	def _make_head_layer(self):
		image_channels = self.cfg.get("image_channels", 3)
		self.cur_channels = self.cfg.get("head_out_channels", 16)

		self.head = nn.Sequential(
			nn.Conv2d(
				in_channels=image_channels,
				out_channels=self.cur_channels,
				kernel_size=3,
				stride=2,
				padding=auto_pad(3, None, 1)
			),
			nn.BatchNorm2d(self.cur_channels),
			nn.Hardswish(),
		)

		self.model_conf_dict["head"] = {"in": image_channels, "out": self.cur_channels}

	def _make_body_layers(self):
		self.layer_1, out_channels = self._make_layer(
			configure=self.cfg["layer1"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer1"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_2, out_channels = self._make_layer(
			configure=self.cfg["layer2"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer2"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_3, out_channels = self._make_layer(
			configure=self.cfg["layer3"],
			input_channel=self.cur_channels,
		)
		self.model_conf_dict["layer3"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_4, out_channels = self._make_layer(
			configure=self.cfg["layer4"],
			input_channel=self.cur_channels,
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

		self.layer_5, out_channels = self._make_layer(
			configure=self.cfg["layer5"],
			input_channel=self.cur_channels,
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": self.cur_channels, "out": out_channels}
		self.cur_channels = out_channels

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
				padding=auto_pad(1, None, 1)
			),
			nn.BatchNorm2d(last_channels),
			nn.Hardswish(),
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
		x = self.head(x)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		x = self.layer_5(x)
		x = self.tail(x)
		x = self.classifier(x)

		return x