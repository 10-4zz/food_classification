import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="hard_sigmoid")
class HardSigmoid(nn.Hardswish):
    def __init__(
            self,
            inplace: bool = False
    ) -> None:
        super(HardSigmoid, self).__init__(inplace=inplace)
