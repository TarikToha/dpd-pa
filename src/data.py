import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class IQDatasetPA(Dataset):
    def __init__(self, root, target_dir, mode, seq_len=0):
        self.mode = mode

        data_in = pd.read_csv(f'{root}/{target_dir}_input.csv')
        data_out = pd.read_csv(f'{root}/{target_dir}_output.csv')

        x = np.column_stack((data_in['I'], data_in['Q']))  # [N, 2]
        y = np.column_stack((data_out['I'], data_out['Q']))

        self.samples = []
        if mode == 'window':
            for i in range(len(x) - seq_len):
                self.samples.append((x[i:i + seq_len], y[i:i + seq_len]))

        else:
            for x_i, y_i in zip(x, y):
                self.samples.append((x_i, y_i))

        print('using', len(self.samples), f'{mode} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        src = torch.tensor(src, dtype=torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float32)

        if self.mode == 'single':
            src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)

        return src, tgt


class IQDatasetDPD(Dataset):
    def __init__(self, root, target_dir, mode, seq_len=0):
        self.mode = mode

        data_in = pd.read_csv(f'{root}/{target_dir}_input.csv')
        data_out = pd.read_csv(f'{root}/{target_dir}_output.csv')

        x = np.column_stack((data_in['I'], data_in['Q']))  # [N, 2]
        y = np.column_stack((data_out['I'], data_out['Q']))

        self.gain = self.calculate_gain(x[:, 0], x[:, 1], y[:, 0], y[:, 1])
        y *= self.gain

        self.samples = []
        if mode == 'window':
            for i in range(len(x) - seq_len):
                self.samples.append((x[i:i + seq_len], y[i:i + seq_len]))

        else:
            for x_i, y_i in zip(x, y):
                self.samples.append((x_i, y_i))

        print('using', len(self.samples), f'{mode} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        src = torch.tensor(src, dtype=torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float32)

        if self.mode == 'single':
            src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)

        return src, tgt

    def calculate_gain(self, in_I, in_Q, out_I, out_Q):
        pa_in = in_I + 1j * in_Q
        pa_out = out_I + 1j * out_Q
        max_in_amp = np.abs(pa_in).max()
        max_out_amp = np.abs(pa_out).max()
        gain = max_out_amp / max_in_amp
        return gain


class PADataModule(pl.LightningDataModule):
    def __init__(
            self,
            root: str = "data/",
            batch_size: int = 32,
            workers: int = 4,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset. One of [custom, cifar10, cifar100, flowers102
                     food101, pets37, stl10, dtd, aircraft, cars]
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            num_classes: Number of classes when using a custom dataset
            size: Crop size
            min_scale: Min crop scale
            max_scale: Max crop scale
            flip_prob: Probability of applying horizontal flip
            rand_aug_n: RandAugment number of augmentations
            rand_aug_m: RandAugment magnitude of augmentations
            erase_prob: Probability of applying random erasing
            use_trivial_aug: Use TrivialAugment instead of RandAugment
            mean: Normalization means
            std: Normalization standard deviations
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = IQDatasetPA(
                root=self.root, target_dir="train", mode='window', seq_len=128
            )

        self.val_dataset = IQDatasetPA(
            root=self.root, target_dir="val", mode='single'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class DPDDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root: str = "data/",
            batch_size: int = 32,
            workers: int = 4,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset. One of [custom, cifar10, cifar100, flowers102
                     food101, pets37, stl10, dtd, aircraft, cars]
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            num_classes: Number of classes when using a custom dataset
            size: Crop size
            min_scale: Min crop scale
            max_scale: Max crop scale
            flip_prob: Probability of applying horizontal flip
            rand_aug_n: RandAugment number of augmentations
            rand_aug_m: RandAugment magnitude of augmentations
            erase_prob: Probability of applying random erasing
            use_trivial_aug: Use TrivialAugment instead of RandAugment
            mean: Normalization means
            std: Normalization standard deviations
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = IQDatasetDPD(
                root=self.root, target_dir="train", mode='window', seq_len=128
            )

        self.val_dataset = IQDatasetDPD(
            root=self.root, target_dir="val", mode='single'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )
