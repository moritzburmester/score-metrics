import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import make_moons
import lightning_data_modules.utils as utils


class TwoMoonsDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        n_samples = config.data.get('data_samples')
        noise = config.data.get('noise_std', 0.05)

        X, _ = make_moons(n_samples=n_samples, noise=noise)
        X = torch.tensor(X, dtype=torch.float32)

        self.data = X

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@utils.register_lightning_datamodule(name='TwoMoons')
class TwoMoonsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split = config.data.split

        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None):
        self.dataset = TwoMoonsDataset(self.config)
        l = len(self.dataset)

        self.train_data, self.valid_data, self.test_data = random_split(
            self.dataset,
            [
                int(self.split[0] * l),
                int(self.split[1] * l),
                int(self.split[2] * l),
            ],
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
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch,
            num_workers=self.test_workers,
        )