from torch import Tensor
import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="prelu")
class PReLU(nn.Module):
    def __init__(
            self,
            num_parameters: int = 1,
            init: float = 0.25,

    ) -> None:
        super(PReLU, self).__init__()
        self.act = nn.PReLU(
            num_parameters=num_parameters,
            init=init,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)
