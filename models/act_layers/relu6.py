import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="relu6")
class ReLU6(nn.ReLU6):
    def __init__(
            self,
            inplace: bool = False,
            **kwargs,
    ) -> None:
        super(ReLU6, self).__init__(inplace=inplace)
