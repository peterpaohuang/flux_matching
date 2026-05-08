import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from anndata import AnnData

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset,
        shuffle,
        validation_split,
        num_workers,
        batch_size,
        collate_fn=default_collate,
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class VeloDataset(Dataset):
    def __init__(
        self,
        data_source,
        train=True,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        if isinstance(data_source, str):
            data_source = Path(data_source)
            with open(data_source, "rb") as f:
                adata = pickle.load(f)
        elif isinstance(data_source, AnnData):
            adata = data_source
        else:
            raise ValueError("data_source must be a file path or anndata object")

        self.Ux_sz = adata.layers["Mu"]
        self.Sx_sz = adata.layers["Ms"]

        if velocity_genes:
            self.Ux_sz = self.Ux_sz[:, adata.var["velocity_genes"]]
            self.Sx_sz = self.Sx_sz[:, adata.var["velocity_genes"]]

        if use_scaled_u:
            scaling = np.std(self.Ux_sz, axis=0) / np.std(self.Sx_sz, axis=0)
            self.Ux_sz = self.Ux_sz / scaling

        N_cell, N_gene = self.Sx_sz.shape

        mask = np.ones([N_cell, N_gene])
        mask[self.Ux_sz == 0] = 0
        mask[self.Sx_sz == 0] = 0

        self.Ux_sz = torch.tensor(self.Ux_sz, dtype=torch.float32)
        self.Sx_sz = torch.tensor(self.Sx_sz, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.Ux_sz)

    def __getitem__(self, i):
        return {
            "idx": i,
            "Ux_sz": self.Ux_sz[i],
            "Sx_sz": self.Sx_sz[i],
            "mask": self.mask[i],
        }


class VeloDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_source,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        batch_size=256,
        training=True,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        self.data_source = data_source
        self.dataset = VeloDataset(
            data_source,
            train=training,
            velocity_genes=velocity_genes,
            use_scaled_u=use_scaled_u,
        )
        self.shuffle = shuffle
        super().__init__(
            self.dataset, shuffle, validation_split, num_workers, batch_size
        )
