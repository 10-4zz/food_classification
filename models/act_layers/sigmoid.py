import torch.nn as nn

from . import ACT_LAYERS


@ACT_LAYERS.register(component_name="sigmoid")
class Sigmoid(nn.Sigmoid):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super(Sigmoid, self).__init__()
