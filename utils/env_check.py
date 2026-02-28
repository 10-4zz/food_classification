# --------------------------------------------------------
# Food Classification
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ian Zhu
# --------------------------------------------------------
import os
import torch
import platform
import sys
from pathlib import Path
from typing import Dict

# Use the modern importlib.metadata to get package versions
from importlib import metadata as importlib_metadata

from utils import logger


TORCH_VERSION = "2.5.0+cu118"
TORCH_CHAUDIO_VERSION = "2.5.0+cu118"
TORCH_VISION_VERSION = "0.20.0+cu118"


def log_environment_info():
    """
    get and log thr information of environment.
    """
    logger.info("=" * 25 + " Environment Information " + "=" * 25)

    logger.info(f"{'Operating System':<25}: {platform.system()} {platform.release()}")

    logger.info(f"{'Python Version':<25}: {sys.version.split(' ')[0]}")

    logger.info(f"{'PyTorch Version':<25}: {torch.__version__}")

    cpu_count = os.cpu_count()
    logger.info(f"{'CPU Count':<25}: {cpu_count}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"{'CUDA Available':<25}: True")
        logger.info(f"{'GPU Count':<25}: {gpu_count}")

        logger.info(f"{'PyTorch Built with CUDA':<25}: {torch.version.cuda}")

        if torch.backends.cudnn.is_available():
            logger.info(f"{'cuDNN Version':<25}: {torch.backends.cudnn.version()}")
        else:
            logger.info(f"{'cuDNN Version':<25}: Not Available")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"- GPU {i}: {gpu_name}, Total Memory: {total_mem:.2f} GB")
    else:
        logger.info(f"{'CUDA Available':<25}: False (PyTorch is in CPU-only mode)")


def parse_requirements() -> Dict[str, str]:
    """
    Parses a requirements.txt file to extract packages with exact version pins.
    Returns:
        A dictionary mapping package names to their required versions.
        e.g., {'torch': '1.13.1', 'numpy': '1.23.5'}
    """
    file_path = "requirements.txt"
    requirements = {}
    if not Path(file_path).is_file():
        logger.error(f"Requirements file not found at: {file_path}")
        return requirements

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue

            # We are only interested in exact version matches (e.g., package==1.2.3)
            if '==' in line:
                try:
                    package, version = line.split('==')
                    requirements[package.strip()] = version.strip()
                except ValueError:
                    logger.warning(f"Could not parse line in requirements file: {line}")

    requirements['torch'] = TORCH_VERSION
    requirements['torchaudio'] = TORCH_CHAUDIO_VERSION
    requirements['torchvision'] = TORCH_VISION_VERSION
    return requirements


def check_dependencies() -> bool:
    """
    Checks if installed packages match the required versions.
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    requirements  = parse_requirements()
    logger.info("Checking package versions...")
    all_ok = True

    for package, required_version in requirements.items():
        try:
            # Get the version of the currently installed package
            installed_version = importlib_metadata.version(package)

            # Check if the installed version matches the required version
            if installed_version != required_version:
                all_ok = False
                # Format the warning message exactly as requested
                warning_msg = (
                    f"Package '{package}' version is {installed_version}, "
                    f"but version {required_version} is required. "
                    "This is okay, but may cause version conflicts."
                )
                logger.warning(warning_msg)

        except importlib_metadata.PackageNotFoundError:
            all_ok = False
            logger.error(f"Required package '{package}' is not installed.")

    if all_ok:
        logger.info("All specified package versions are installed and correct.")

    return all_ok


def check_cuda_torch_compatibility():
    """
    Checks the compatibility between the system's NVIDIA driver CUDA version
    and the CUDA version PyTorch was compiled with.
    """
    logger.info("Checking cuda versions...")

    # --- System and Python Info ---
    logger.info(f"{'Operating System':<28}: {platform.system()} {platform.release()}")
    logger.info(f"{'Python Version':<28}: {sys.version.split(' ')[0]}")

    # --- PyTorch Info ---
    logger.info(f"{'PyTorch Version':<28}: {torch.__version__}")

    if not torch.cuda.is_available():
        logger.info("❌ PyTorch CUDA environment not found!")
        logger.info("Your PyTorch installation is likely in CPU-only version, or your device don't have cuda. Please check. ")
        return

    torch_cuda_version = torch.version.cuda
    logger.info(f"{'PyTorch Compiled with CUDA':<28}: {torch_cuda_version}")

    # --- System Driver Info ---
    try:
        import pynvml
        pynvml.nvmlInit()

        driver_version_str = pynvml.nvmlSystemGetDriverVersion()
        cuda_driver_version_int = pynvml.nvmlSystemGetCudaDriverVersion_v2()

        driver_cuda_major = cuda_driver_version_int // 1000
        driver_cuda_minor = (cuda_driver_version_int % 1000) // 10

        logger.info(f"{'System NVIDIA Driver Version':<28}: {driver_version_str}")
        logger.info(f"{'Driver Supports up to CUDA':<28}: {driver_cuda_major}.{driver_cuda_minor}")

        pynvml.nvmlShutdown()

    except Exception as e:
        logger.info("❌ Could not retrieve NVIDIA driver information.")
        logger.info(f"Error: {e}")
        logger.info("Please ensure the NVIDIA driver is correctly installed and ")
        logger.info("the 'nvidia-ml-py' library is installed (`pip install nvidia-ml-py`).")
        return

    # --- Comparison and Conclusion ---
    torch_cuda_major = int(torch_cuda_version.split('.')[0])
    torch_cuda_minor = int(torch_cuda_version.split('.')[1])

    if (driver_cuda_major > torch_cuda_major) or \
            (driver_cuda_major == torch_cuda_major and driver_cuda_minor >= torch_cuda_minor):
        logger.info("✅ Compatibility Check Passed!")
        logger.info("Your NVIDIA driver version is sufficient for the CUDA runtime version used by PyTorch.")
    else:
        logger.info("❌ Compatibility Check Failed!")
        logger.info("Reason: Your PyTorch requires CUDA runtime support for version "
              f"{torch_cuda_major}.{torch_cuda_minor},")
        logger.info("but your installed NVIDIA driver only supports up to CUDA version "
              f"{driver_cuda_major}.{driver_cuda_minor}.")
        logger.info("Solution: Please update your NVIDIA graphics driver to the latest version.")


if __name__ == '__main__':
    log_environment_info()

    # Create a dummy requirements.txt for the example
    dummy_requirements_content = """
        # This is a comment, it will be ignored
        torch==1.13.1  # An example of a correct version (assuming you have this)
        numpy==99.9.9   # This version will definitely not match
        requests        # This line will be ignored because it doesn't use '=='
        non_existent_package==1.0.0 # This package is likely not installed
    """
    with open("requirements.txt", "w") as f:
        f.write(dummy_requirements_content)

    # Run the check
    check_dependencies()

    # Clean up the dummy file
    os.remove("requirements.txt")


