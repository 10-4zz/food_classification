from typing import Dict, List
import os
import importlib
from pathlib import Path

import warnings


class Registry:
    def __init__(
            self,
            registry_name: str,
            component_dir: List[str] = None,
    ) -> None:
        """
            Args:
                registry_name (str): The name of the registry.
                component_dir (List[str], optional): The directory to load the components from.
        note: The component_dir should be a list of directory paths where the components are located,
        if it is None, the error will be occurred.
        """
        self.registry_name: str = registry_name
        if component_dir is None:
            warnings.warn("This operation may lead to issues when loading components from package.")
        self.component_dir = component_dir
        self._registry: Dict = {}

        self._loaded = False

    def info(self) -> str:
        return f"Registry: {self.registry_name}, Number of items: {len(self._registry)}"

    def _load_components(self) -> None:
        if self._loaded or not self.component_dir:
            return
        self._loaded = True
        for dir_path in self.component_dir:
            if not os.path.isdir(dir_path):
                continue
            # Traverse all .py files in the directory (excluding __init__.py)
            if "/" in dir_path:
                package = dir_path.replace("/", ".")
            else:
                package = dir_path
            for filename in os.listdir(dir_path):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]  # Remove the .py suffix
                    # Import the module (assuming the directory is in the Python path)
                    importlib.import_module(f"{package}.{module_name}")

    def register(self, component_name: str, another_name: str = None):
        def register_component(component):
            name = component_name
            if name is None:
                warnings.warn("Component name is None, using the component's class name instead.")
                name = component.__name__
            if name in self._registry:
                raise ValueError(f"Component '{name}' is already registered in {self.registry_name}.")
            self._registry[name] = component
            if another_name is not None:
                if another_name in self._registry:
                    raise ValueError(f"Component '{another_name}' is already registered in {self.registry_name}.")
                self._registry[another_name] = component
            return component
        return register_component

    def get(self, component_name: str):
        self._load_components()
        if component_name not in self._registry:
            raise KeyError(f"Component '{component_name}' is not registered in {self.registry_name}.")
        return self._registry[component_name]

    def get_keys(self):
        self._load_components()
        return self._registry.keys()

    def get_values(self):
        self._load_components()
        return self._registry.values()

