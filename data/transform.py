# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Iran Zhu
# --------------------------------------------------------

from typing import Any

from torchvision import transforms


def build_transform(
    is_train: bool = True,
    config: Any = None,
):
    if is_train:
        return transforms.Compose([transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(config.DATA.IMG_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
