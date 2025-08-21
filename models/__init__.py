from utils import Registry


MODEL_REGISTRIES = Registry(
    registry_name="model_registry",
    component_dir=["models"]
)


def build_model(config):
    """
    Build a model based on the provided model name and additional keyword arguments.

    Args:
        config: Configuration object containing model settings.

    Returns:
        An instance of the specified model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    create_fn = MODEL_REGISTRIES.get(config.MODEL.NAME)
    model = create_fn(config, config.MODEL.NUM_CLASSES)

    return model
