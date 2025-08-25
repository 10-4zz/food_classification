# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Iran Zhu
# --------------------------------------------------------

import numpy as np

import logging

import torch

from .dataset import build_dataset
from .sampler import SubsetRandomSampler
from .mixup import Mixup


def build_loader(config, logger: logging.Logger, transform=None):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True,
        config=config,
        logger=logger,
        transform=transform['train'] if transform is not None else None
    )
    config.freeze()
    print(f"Successfully build train dataset")
    dataset_val, _ = build_dataset(
        is_train=False,
        config=config,
        logger=logger,
        transform=transform['val'] if transform is not None else None
    )
    print(f"Successfully build val dataset")

    indices = np.arange(len(dataset_train))
    sampler_train = SubsetRandomSampler(indices)


    indices = np.arange(len(dataset_val))
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn