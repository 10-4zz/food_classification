import warnings

from utils import Registry


ACT_LAYERS = Registry(
    registry_name='act_layers_registry',
    component_dir=["models/act_layers"]
)


def get_act_layers(act_name: str = 'relu', **kwargs):
    """
    Get the activation layers by name.
    """
    act_name = act_name.lower()
    if act_name not in ACT_LAYERS.get_keys():
        warnings.warn("Activation layer not found, using 'relu' as default.")
        return ACT_LAYERS.get(component_name='relu')(**kwargs)
    else:
        return ACT_LAYERS.get(component_name=act_name)(**kwargs)
