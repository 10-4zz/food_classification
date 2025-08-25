from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="sigmoid")
class Sigmoid(nn.Module):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super(Sigmoid, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
