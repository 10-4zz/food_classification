# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Iran Zhu
# --------------------------------------------------------

import logging
from copy import deepcopy
from typing import Callable, Any
import time

import thop

try:
    import torch
except:
    raise ImportError("Please install torch package to use this module, "
                      "or you set a value to the virtual_input."
                      "eg. torch.empty((1, 3, 224, 224), device=device)")


def summery_model(
    model: torch.nn.Module,
    logger: logging.Logger,
    input_virtual: Any = None,
    image_size: int = 224,
    unit_param: str = "M",
    unit_macs: str = "M",
    unit_flops: str = "M",
):
    """
    Args:
        model (torch.nn.Module): The model to summarize.
        logger (logging.Logger): Logger to log the summary.
        input_virtual (Any, optional): A virtual input tensor for the model. Defaults to None.
        image_size (int, optional): The size of the input image. Defaults to 224.
        unit_param (str, optional): Unit for parameters ('K', 'M', 'G'). Defaults to 'M'.
        unit_macs (str, optional): Unit for MACs ('K', 'M', 'G'). Defaults to 'M'.
        unit_flops (str, optional): Unit for FLOPs ('K', 'M', 'G'). Defaults to 'M'.
    """
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if unit_param == "K":
        n_parameters /= 1e3
    elif unit_param == "M":
        n_parameters /= 1e6
    elif unit_param == "G":
        n_parameters /= 1e9
    else:
        logger.error(f"Invalid unit for parameters: {unit_param}. It just supports 'K', 'M', or 'G'.")
    param_content = "{:<18}={:>8.3f}{}".format("Overall parameters", n_parameters, unit_param)
    logger.info(param_content)

    p = next(model.parameters())
    if input_virtual is not None:
        im = input_virtual
    else:
        im = torch.empty((1, 3, image_size, image_size), device=p.device)

    macs = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0]
    flops = macs * 2
    if unit_macs == "K":
        macs /= 1e3
    elif unit_macs == "M":
        macs /= 1e6
    elif unit_macs == "G":
        macs /= 1e9
    else:
        logger.error(f"Invalid unit for macs: {unit_macs}. It just supports 'K', 'M', or 'G'.")
    macs_content = "{:<20}={:>8.1f}{}".format("Overall MACs (Thop)**", macs, unit_macs)
    logger.info(macs_content)

    if unit_flops == "K":
        flops /= 1e3
    elif unit_flops == "M":
        flops /= 1e6
    elif unit_flops == "G":
        flops /= 1e9
    else:
        logger.error(f"Invalid unit for macs: {unit_flops}. It just supports 'K', 'M', or 'G'.")
    flops_content = "{:<20}={:>8.1f}{}".format("Overall FLOPs (Thop)**", flops, unit_flops)
    logger.info(flops_content)


@torch.no_grad()
def throughput(
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        logger: logging.Logger,
):
    """
    Args:
        data_loader (torch.utils.data.DataLoader): The data loader for the model.
        model (torch.nn.Module): The model to evaluate throughput.
        logger (logging.Logger): Logger to log the throughput.
    """
    model.eval()

    for _, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
