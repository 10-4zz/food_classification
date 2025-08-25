from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="hard_sigmoid")
class HardSigmoid(nn.Module):
    def __init__(
            self,
            inplace: bool = False,
    ) -> None:
        super(HardSigmoid, self).__init__()
        self.act = nn.Hardsigmoid(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
