from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="leaky_relu")
class LeakyReLU(nn.Module):
    def __init__(
            self,
            negative_slope: float = 1e-2,
            inplace: bool = False,
    ) -> None:
        super(LeakyReLU, self).__init__()
        self.act = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=inplace
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)