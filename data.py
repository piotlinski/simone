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
import os
# Turn off annoying tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import urllib.request
from os.path import exists
from multiprocessing.pool import ThreadPool
import os

from quarantine.zack.simone import cater_with_masks
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
from einops import rearrange
import pytorch_lightning as pl
import torch.distributed


def fetch_dataset(local_dir="dataset/"):
    """Download the dataset to `local_dir`."""
    # It might be possible to create a tfrecords dataset directly from the GCP URLs instead of downloading locally?
    def download_file(x):
        dataset, i = x
        url = f"https://storage.googleapis.com/multi-object-datasets/cater_with_masks/cater_with_masks_{dataset}.tfrecords-00{i:03d}-of-00100"
        # print(url)
        local_path = f"{local_dir}/{dataset}/{i}.tfrecord"

        if not exists(local_path):
            urllib.request.urlretrieve(url, local_path)
        return url

    urls = []
    for dataset in ("train", "test"):
        if not os.path.exists(f"{local_dir}/{dataset}"):
            os.makedirs(f"{local_dir}/{dataset}")
        for i in range(100):
            urls.append((dataset, i))
    ThreadPool(8).imap_unordered(download_file, urls)


def yield_item_from_deepmind_dataset(dataset, sample_limit=None):
    # batch_dataset = dataset.batch(batch_size)
    for i, item in enumerate(dataset):
        if sample_limit and i >= sample_limit:
            break
        video = item["image"][:16]
        mask = item["mask"][:16]
        video = (torch.tensor(video.numpy()) / 255)
        video = rearrange(video, "t h w c -> t c h w")
        # shape b, T, n_objects, w, h, 1
        mask = torch.tensor(mask.numpy())
        yield video, mask


def tfrecords_paths(mode="train", local_dir="dataset/"):
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


class CaterWithMasks(IterableDataset):
    def __init__(self, world_size, mode="train", sample_limit=None):
        # Sample limit will apply to each worker separately
        self.sample_limit = sample_limit
        self.world_size = world_size
        self.mode = mode

    def __iter__(self):
        ddp_idx = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp_idx = ddp_idx
        n_ddp = self.world_size
        all_paths = tfrecords_paths(self.mode)
        num_files = len(all_paths) // n_ddp
        self.this_gpu_paths = all_paths[ddp_idx * num_files: (ddp_idx + 1) * num_files]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            our_paths = self.this_gpu_paths
            print("single process loading")
        else:  # in a worker process
            idx = worker_info.id
            # print(f"loading from {self.ddp_idx}, {idx} worker")
            num_workers = worker_info.num_workers
            num_files = len(self.this_gpu_paths) // num_workers
            our_paths = self.this_gpu_paths[idx * num_files: (idx + 1) * num_files]
        tf_dataset = cater_with_masks.dataset(our_paths)

        return iter(yield_item_from_deepmind_dataset(tf_dataset, sample_limit = self.sample_limit))


class CATERDataModule(pl.LightningDataModule):
    def __init__(self, n_gpus, train_batch_size, val_batch_size, val_dataset_size, num_workers=4):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_dataset_size = val_dataset_size
        self.n_gpus = n_gpus

    def setup(self, stage = None):
        self.train_dataset = CaterWithMasks(self.n_gpus, "train")
        self.val_dataset = CaterWithMasks(self.n_gpus, "test", sample_limit=self.val_dataset_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )


def get_datamodule(n_gpus, train_batch_size, val_batch_size, val_dataset_size, num_workers):
    return CATERDataModule(n_gpus, train_batch_size, val_batch_size, val_dataset_size, num_workers)
