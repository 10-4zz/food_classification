from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="relu")
class ReLU(nn.Module):
    def __init__(
            self,
            inplace: bool = False,
    ) -> None:
        super(ReLU, self).__init__()
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
