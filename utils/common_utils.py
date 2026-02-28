# --------------------------------------------------------
# Food Classification
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ian Zhu
# --------------------------------------------------------
import argparse
from typing import Tuple
import warnings
from yacs.config import CfgNode

import random
import numpy as np
import torch

from utils import logger


def set_random_seed(config: CfgNode) -> None:
    """
    Set random seed for reproducibility.
    :param opts:
    :return:
    """
    logger.info("=" * 20 + " Setting Random Seed " + "=" * 20)
    seed = config.COMMON.SEED
    deterministic = config.COMMON.CUDNN_DETERMINISTIC
    benchmark = config.COMMON.CUDNN_DBENCHMARK
    logger.info("Set random seed to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic, benchmark = check_deterministic_and_benchmark(deterministic, benchmark)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    logger.info("cudnn.deterministic set to {}, cudnn.benchmark set to {}".format(deterministic, benchmark))
    if deterministic and not benchmark:
        logger.info("Note: This setting may lead to slow training speed, but ensures reproducibility.")
    else:
        logger.info("Note: This setting may lead to non-deterministic results, but improves training speed.")
    logger.info("Random seed set successfully.")


def check_deterministic_and_benchmark(deterministic: bool, benchmark: bool) -> Tuple[bool, bool]:
    """
    Check the current settings for cudnn deterministic and benchmark.
    :return:
    """
    if deterministic:
        if benchmark:
            logger.warning("cudnn.deterministic is True, but cudnn.benchmark is also True. "
                           "This may lead to non-deterministic results. The cudnn.benchmark will be set to False.")
            benchmark = False
    else:
        if not benchmark:
            logger.warning("cudnn.deterministic is False, and cudnn.benchmark is also False. "
                           "This may lead to slow training. The cudnn.benchmark will be set cudnn.benchmark"
                           " to True for better performance.")
            benchmark = True

    return deterministic, benchmark


def filter_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd` is deprecated", category=FutureWarning)
    warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_bwd` is deprecated", category=FutureWarning)
