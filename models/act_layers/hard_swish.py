import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="hard_swish")
class HardSwish(nn.Hardswish):
    def __init__(
            self,
            inplace: bool = False
    ) -> None:
        super(HardSwish, self).__init__(inplace=inplace)
