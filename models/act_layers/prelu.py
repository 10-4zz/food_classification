from typing import Any

import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="prelu")
class PReLU(nn.PReLU):
    def __init__(
            self,
            num_parameters: int = 1,
            init: float = 0.25,
            device: Any = None,
            dtype: Any = None
    ) -> None:
        super(PReLU, self).__init__(
            num_parameters=num_parameters,
            init=init,
            device=device,
            dtype=dtype
        )
