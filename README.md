
A Pytorch implementation of [SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition](https://deepmind.com/research/publications/2021/SIMONe-View-Invariant-Temporally-Abstracted-Object-Representations-via-Unsupervised-Video-Decomposition)

For an explanation of the model and code, see our [blog post](https://generallyintelligent.ai/open-source/2022-04-14-simone/).

Currently, only the [CATER](https://github.com/deepmind/multi_object_datasets) dataset has been implemented.

## Usage

The CATER dataset should download automatically on the first run of `train.py`. `batch_size` is per-GPU, so the total batch size is `batch_size * gpus`.

See `train.py --help` for valid flags.

Small model:
`python train.py --batch_size 4 --learning_rate .0002 --gpus 4 --transformer_layers 1 --max_epochs -1`

Full model from the paper:
`python train.py --batch_size 32 --learning_rate .0002 --gpus 8 --transformer_layers 4 --max_epochs -1`

## Results

Small model:
- trains in ~10 hours (to 100 epochs) on 4x 3090s
- ~.95 ARI on CATER
- [Weights and Biases run metrics](https://wandb.ai/sourceress/simone_public/runs/qataicib?workspace=user-zplizzi)

## Installing

Tested on Ubuntu 20.04; Python 3.9.2

`pip install -r requirements.txt`

For full reproducibility, use the fully pinned package list:
`pip install -r requirements_pinned.txt`
