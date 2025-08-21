import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="gelu")
class GeLU(nn.GELU):
    def __init__(
            self,
            approximate: str = "none",
            **kwargs,
    ) -> None:
        super(GeLU, self).__init__(approximate=approximate)
