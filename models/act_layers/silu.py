from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="silu", another_name="swish")
class SiLU(nn.Module):
    def __init__(
            self,
            inplace: bool = False,
            **kwargs,
    ) -> None:
        super(SiLU, self).__init__()
        self.act = nn.SiLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
