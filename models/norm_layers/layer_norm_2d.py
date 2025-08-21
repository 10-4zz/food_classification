import torch.nn as nn

from . import NORM_LAYERS


@NORM_LAYERS.register(component_name='layer_norm_2d')
class LayerNorm2d(nn.GroupNorm):
    def __init__(
            self,
            num_groups: int,
            num_channels: int,
    ):
        super(LayerNorm2d, self).__init__(
            num_groups=num_groups,
            num_channels=num_channels,
        )
