import warnings

import torch.nn as nn

from utils import Registry


NORM_LAYERS = Registry(
    registry_name='norm_layers_registry',
    component_dir=["models/norm_layers"]
)


def get_norm_layers(norm_name: str = 'relu'):
    """
    Get the Normalization layers by name.
    """
    norm_name = norm_name.lower()
    if norm_name not in NORM_LAYERS.get_keys():
        warnings.warn("Normalization layer not found, using 'Identity' as default.")
        return nn.Identity()
    else:
        return NORM_LAYERS.get(component_name=norm_name)
