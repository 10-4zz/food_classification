import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="relu")
class ReLU(nn.ReLU):
    def __init__(
            self,
            inplace: bool = False,
    ) -> None:
        super(ReLU, self).__init__(inplace=inplace)
