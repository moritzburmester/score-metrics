import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import lightning_data_modules.utils as utils

# ----------------------------
# Dataset
# ----------------------------
class GaussianBlobDataset(Dataset):
    """
    Simple 2D Gaussian blob dataset.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.data = self.generate_data(
        config.data.get("data_samples", 5000),
        config.data.get("ambient_dim", 2),
        config.data.get("means"),
        config.data.get("stds"),
        config.data.get("weights"),
    )

    def generate_data(self, n_samples, ambient_dim, means, stds, weights):
        means = torch.tensor(means, dtype=torch.float)  # (K, 2)
        stds = torch.tensor(stds, dtype=torch.float)    # (K,)
        weights = torch.tensor(weights, dtype=torch.float)

        weights = weights / weights.sum()  # normalize

        n_components = len(means)

        # Sample which component each point comes from
        component_ids = torch.multinomial(weights, n_samples, replacement=True)

        data = torch.zeros(n_samples, ambient_dim)

        for k in range(n_components):
            idx = (component_ids == k)
            num_k = idx.sum()
            if num_k > 0:
                data[idx] = means[k] + stds[k] * torch.randn(num_k, ambient_dim)

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# ----------------------------
# DataModule
# ----------------------------
@utils.register_lightning_datamodule(name="GaussianBlob")
class GaussianBlobDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split = config.data.split  # e.g., [0.8, 0.1, 0.1]

        # DataLoader params
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None):
        # Create dataset
        self.dataset = GaussianBlobDataset(self.config)
        l = len(self.dataset)

        # Compute splits and ensure sum equals dataset length
        train_len = int(self.split[0] * l)
        val_len = int(self.split[1] * l)
        test_len = l - train_len - val_len

        self.train_data, self.valid_data, self.test_data = random_split(
            self.dataset, [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch,
            num_workers=self.train_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.val_batch,
            num_workers=self.val_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch,
            num_workers=self.test_workers,
            shuffle=False,
        )