from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="relu6")
class ReLU6(nn.Module):
    def __init__(
            self,
            inplace: bool = False,
            **kwargs,
    ) -> None:
        super(ReLU6, self).__init__()
        self.act = nn.ReLU6(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
