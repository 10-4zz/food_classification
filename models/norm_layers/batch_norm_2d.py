from typing import Optional

import torch.nn as nn

from . import NORM_LAYERS


@NORM_LAYERS.register(component_name='batch_norm_2d', another_name='batch_norm')
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
            self,
            num_features: int,
            momentum: Optional[float] = 0.1,
    ):
        super(BatchNorm2d, self).__init__(
            num_features=num_features,
            momentum=momentum
        )
