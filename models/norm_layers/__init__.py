import warnings

import torch.nn as nn

from utils import Registry


NORM_LAYERS = Registry(
    registry_name='norm_layers_registry',
    component_dir=["models/norm_layers"]
)


def get_norm_layers(
        num_features: int,
        norm_name: str = 'layer_norm_2d',
        num_groups: int = 1,
        momentum: float = 0.1
):
    """
    Get the Normalization layers by name.
    """
    norm_name = norm_name.lower()
    if norm_name not in NORM_LAYERS.get_keys():
        warnings.warn("Normalization layer not found, using 'Identity' as default.")
        return nn.Identity()
    else:
        create_fn = NORM_LAYERS.get(component_name=norm_name)
        norm_layer = create_fn(
            num_features=num_features,
            num_groups=num_groups,
            momentum=momentum
        )
        return norm_layer
