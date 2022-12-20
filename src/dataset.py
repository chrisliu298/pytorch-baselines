import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


class BaseDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MNISTDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def prepare_data(self):
        MNIST(self.cfg.data_dir, train=True, download=True)
        MNIST(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load train and test datasets
        x_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = MNIST(self.cfg.data_dir, train=True, transform=x_transforms)
        self.test_dataset = MNIST(
            self.cfg.data_dir, train=False, transform=x_transforms
        )
        # Split train dataset into train and validation datasets
        indices = torch.randperm(len(train_dataset)).tolist()
        train_idx, val_idx = indices[:-10000], indices[-10000:]
        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(train_dataset, val_idx)


class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def prepare_data(self):
        FashionMNIST(self.cfg.data_dir, train=True, download=True)
        FashionMNIST(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load train and test datasets
        x_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        train_dataset = FashionMNIST(
            self.cfg.data_dir, train=True, transform=x_transforms
        )
        self.test_dataset = FashionMNIST(
            self.cfg.data_dir, train=False, transform=x_transforms
        )
        # Split train dataset into train and validation datasets
        indices = torch.randperm(len(train_dataset)).tolist()
        train_idx, val_idx = indices[:-10000], indices[-10000:]
        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(train_dataset, val_idx)
