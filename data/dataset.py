# --------------------------------------------------------
# Food Classification
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ian Zhu
# --------------------------------------------------------

import os
from typing import Any, Callable, Optional, Tuple

import logging

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import ImageFolder, default_loader

from .transform import build_transform


def build_dataset(is_train, config, logger: logging.Logger, transform=None):
    if transform is None:
        logger.info("Building dataset with default transform.")
        transform = build_transform(is_train, config)
    else:
        logger.info("Building dataset with custom transform.")
        transform = transform
    if config.DATA.DATASET == 'FOOD5':
        root = config.DATA.DATA_PATH
        dataset = Food5DataFolder(root, is_train=is_train, transform=transform)
        nb_classes = 5
    elif config.DATA.DATASET == 'ETHZFOOD101':
        root = config.DATA.DATA_PATH
        dataset = Food101DataFolder(root, is_train=is_train, transform=transform)
        nb_classes = 101
    elif config.DATA.DATASET == 'VIREOFOOD172':
        root = os.path.join(config.DATA.DATA_PATH, 'train' if is_train else 'val')
        dataset = Food172DataFolder(root, transform=transform)
        nb_classes = 172
    elif config.DATA.DATASET == 'UECFOOD256':
        root = os.path.join(config.DATA.DATA_PATH, 'train' if is_train else 'val')
        dataset = Food256DataFolder(root, transform=transform)
        nb_classes = 256
    else:
        raise NotImplementedError("The dataset is not supported: {}".format(config.DATA.DATASET))

    return dataset, nb_classes


class Food5DataFolder(VisionDataset):
    def __init__(self,
                 root: str,
                 is_train: Optional[bool] = True,
                 loader: Callable[[str], Any] = default_loader,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if is_train:
            index_file = 'train.txt'
        else:
            index_file = 'val.txt'

        index_file_handler = open(os.path.join(root, index_file), 'r')
        self.imgs = []
        for line in index_file_handler:
            line = line.strip()
            words = line.split(' ')
            self.imgs.append((os.path.join(root, words[0]), int(words[1])))

        self.loader = loader
        self.samples = self.imgs
        self.targets = [s[1] for s in self.samples]
        self.is_train = is_train

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class Food101DataFolder(VisionDataset):
    def __init__(self,
                 root: str,
                 is_train: Optional[bool] = True,
                 loader: Callable[[str], Any] = default_loader,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if is_train:
            index_file = 'train_full.txt'
        else:
            index_file = 'test_full.txt'

        index_file_handler = open(os.path.join(root, index_file), 'r')
        self.imgs = []
        for line in index_file_handler:
            line = line.strip()
            words = line.split(' ')
            self.imgs.append((os.path.join(root, words[0]), int(words[1])))

        self.loader = loader
        self.samples = self.imgs
        self.targets = [s[1] for s in self.samples]
        self.is_train = is_train

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class Food172DataFolder(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)


class Food256DataFolder(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)