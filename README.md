
A Pytorch implementation of [SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition](https://deepmind.com/research/publications/2021/SIMONe-View-Invariant-Temporally-Abstracted-Object-Representations-via-Unsupervised-Video-Decomposition)

Currently, only the [CATER](https://github.com/deepmind/multi_object_datasets) dataset has been implemented.

## TODOs

- Validate on multiple val batches.
- create a proper requirements.py file

## Usage

The CATER dataset should download automatically on the first run of `train.py`. `batch_size` is per-GPU, so the total batch size is `batch_size * gpus`.

Small model:
`python train.py --batch_size 4 --learning_rate .0002 --gpus 4 --transformer_layers 1`

Full model from the paper:
`python train.py --batch_size 32 --learning_rate .0002 --gpus 8 --transformer_layers 4`

## Results

Small model:
- trains in ~10 hours (to 100 epochs) on 4x 3090s
- ~.95 ARI on CATER
- [Weights and Biases run metrics](https://wandb.ai/sourceress/simone_public/runs/qataicib?workspace=user-zplizzi)

## Dependencies

(non-exhaustive list)

- pytorch 1.10
- tensorflow CPU-only (the GPU interferes with pytorch for some reason on my machine): https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.7.0-cp39-cp39-manylinux2010_x86_64.whl
- einops
- wandb[service]
- this version of pytorch-lightning: git+https://github.com/wandb/pytorch-lightning.git@wandb-service-attach
