"""
# Their dataset
# train has ~40k videos of length 16 frames
# test has ~17k videos

This file loads in the CATER dataset that includes ground truth segmentation masks,
released here: https://github.com/deepmind/multi_object_datasets/

Unfortunately loading a TFRecords dataset into pytorch in a way that's friendly to DDP is really not cleanly doable.
This is because TFRecords doesn't allow random access, only sequential access.

However, the dataset is sharded into 100 files, and each can be loaded independently.
The approach taken here is to allocate 100//n_gpus shards to each GPU,
and then shards_per_gpu//num_workers to each worker, to allow multiple workers per GPU.

This isn't ideal because n_gpus*n_workers in general won't divide evenly into 100,
so we just won't load some shards. And 2 of the training shards don't have an even number of examples,
so we just excluded those entirely since we need each worker to have an identical number of examples.
"""
import json
import os
import urllib.request
from multiprocessing.pool import ThreadPool
from os.path import exists
from typing import Optional
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from contrib.deepmind import cater_with_masks

# Turn off annoying tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def download_dataset(local_dir: str = "dataset/"):
    """Download the dataset to `local_dir`."""
    # It might be possible to create a tfrecords dataset directly from the GCP URLs instead of downloading locally?
    return
    def download_file(x):
        dataset, i = x
        url = f"https://storage.googleapis.com/multi-object-datasets/cater_with_masks/cater_with_masks_{dataset}.tfrecords-00{i:03d}-of-00100"
        # print(url)
        local_path = f"{local_dir}/{dataset}/{i}.tfrecord"

        if not exists(local_path):
            urllib.request.urlretrieve(url, local_path)
            print(f"downloaded {url} to {local_path}")
        return url

    urls = []
    for dataset in ("train", "test"):
        if not os.path.exists(f"{local_dir}/{dataset}"):
            os.makedirs(f"{local_dir}/{dataset}")
        for i in range(100):
            urls.append((dataset, i))
    out = ThreadPool(8).imap_unordered(download_file, urls)
    for result in out:
        pass


def _yield_item_from_deepmind_dataset(dataset, sample_limit=None):
    # batch_dataset = dataset.batch(batch_size)
    for i, item in enumerate(dataset):
        if sample_limit and i >= sample_limit:
            break
        video = item["image"][:16]
        mask = item["mask"][:16]
        video = torch.tensor(video.numpy()) / 255
        video = rearrange(video, "t h w c -> t c h w")
        # shape b, T, n_objects, w, h, 1
        mask = torch.tensor(mask.numpy())
        yield video, mask


def _tfrecords_paths(mode: str = "train", local_dir: str = "dataset/"):
    """Get a list of the tfrecord file paths."""
    assert mode in ("train", "test")
    paths = []
    for i in range(100):
        # train: 21 and 74 have abnormal numbers of samples (31 and 231, respectively). All others have 399.
        # Exclude these so all workers have the same number of examples.
        # test: all files are good, with 171 videos.
        if mode == "train":
            if i in (21, 74):
                continue
        paths.append(f"{local_dir}/{mode}/{i}.tfrecord")
    return paths


class MultiScaleMNIST(Dataset):
    def __init__(self, world_size: int, mode: str = "train", sample_limit: int = None):
        """
        Args:
            world_size: the number of DDP processes
            mode: "train" or "test"
            sample_limit: the max number of samples to use from the dataset. It applies to each worker separately.
        """
        super().__init__()
        self.sample_limit = sample_limit
        self.world_size = world_size
        self.mode = mode

        path = os.path.join("dataset/", mode)

        self.sequences = list(sorted(Path(path).glob("*")))
        self.anns = [json.load(p.joinpath("annotations.json").open()) for p in self.sequences]

    @staticmethod
    def load_image(path):
        return np.array(Image.open(path))

    def __getitem__(self, index):
        sequence = self.sequences[index]

        images_files = list(sorted(sequence.glob("*.jpg")))
        imgs = np.stack(self.load_image(p) for p in images_files).transpose(0, 3, 1, 2)

        imgs = imgs.astype(np.float) / 255.0
        imgs = torch.from_numpy(imgs).float()

        return imgs, imgs.expand(imgs.shape[0], 10, *imgs.shape[1:])

    def __len__(self):
        return len(self.sequences)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_gpus: int,
        train_batch_size: int,
        val_batch_size: int,
        val_dataset_size: int,
        num_train_workers: int = 4,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_train_workers = num_train_workers
        self.val_dataset_size = val_dataset_size
        self.n_gpus = n_gpus

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MultiScaleMNIST(self.n_gpus, "train")
        self.val_dataset = MultiScaleMNIST(self.n_gpus, "test", sample_limit=self.val_dataset_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_train_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )
