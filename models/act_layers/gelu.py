from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="gelu")
class GeLU(nn.Module):
    def __init__(
            self,
            approximate: str = "none",
            **kwargs,
    ) -> None:
        super(GeLU, self).__init__()
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)