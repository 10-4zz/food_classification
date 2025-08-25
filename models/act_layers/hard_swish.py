from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="hard_swish")
class HardSwish(nn.Module):
    def __init__(
            self,
            inplace: bool = False,
    ) -> None:
        super(HardSwish, self).__init__()
        self.act = nn.Hardswish(inplace=inplace)

    def forward(self, x) -> Tensor:
        return self.act(x)
