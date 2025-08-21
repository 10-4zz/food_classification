import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="silu", another_name="swish")
class SiLU(nn.SiLU):
    def __init__(
            self,
            inplace: bool = False,
            **kwargs,
    ) -> None:
        super(SiLU, self).__init__(inplace=inplace)
