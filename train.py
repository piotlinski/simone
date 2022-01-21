import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.distributed

from data import fetch_dataset
from data import get_datamodule
from model import SIMONE
from util import str2bool
from config import LOG_FREQ
from metrics import WandbCallback
import wandb_lib


if __name__ == "__main__":
    torch.set_printoptions(precision=10)

    wandb_run_name = f"{datetime.utcnow().isoformat(timespec='seconds')}"

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--name", type=str, default=wandb_run_name)
    parser.add_argument("--project", type=str, default="zack_simone")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=20e-5)
    parser.add_argument("--beta_o", type=float, default=1e-5)
    parser.add_argument("--sigma_x", type=float, default=0.08)
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--ckpt_runid", type=str, default=None)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--decoder_layer_norm", type=str2bool, default=True)
    parser.add_argument("--decoder_ln_scale_bias", type=str2bool, default=False)
    parser.add_argument("--val_batches", type=int, default=10)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    # If we want to override hyperparams in a checkpoint, we have to specify them here
    kwargs = {
        "learning_rate": args.learning_rate,
        "beta_o": args.beta_o,
        "sigma_x": args.sigma_x,
        "transformer_layers": args.transformer_layers,
        "decoder_layer_norm": args.decoder_layer_norm,
        "decoder_ln_scale_bias": args.decoder_ln_scale_bias,
    }
    # Load checkpoint if specified
    if args.ckpt_file and args.ckpt_runid:
        checkpoint = wandb_lib.download_file(args.ckpt_runid, "zack_simone", f"checkpoints/{args.ckpt_file}")
        model = SIMONE.load_from_checkpoint(checkpoint, **kwargs)
    else:
        model = SIMONE(args, **kwargs)

    # Set up wandb logging
    pl_logger = WandbLogger(
        project=args.project,
        name=args.name,
        offline=False,
        log_model=True,  # upload checkpoints to wandb
    )
    # Log gradient and weight histograms
    pl_logger.watch(model, log="all", log_freq=LOG_FREQ * 10)
    # Fix checkpoint path so wandb uploads them properly
    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["EXP_LOG_DIR"] = pl_logger.experiment.dir

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.environ["EXP_LOG_DIR"], "checkpoints"),
        ),
        WandbCallback(),
    ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=pl_logger,
        precision=16,
        strategy="ddp",
        gradient_clip_val=5,
        max_epochs=-1,
    )
    # Something the wandb folks said I needed to include
    wandb.require(experiment="service")

    # Set up the dataset
    fetch_dataset()
    # You'll need to tune this to fit on your GPU, probably.
    val_batch_size = 4
    num_workers = 4
    datamodule = get_datamodule(
        n_gpus=args.gpus,
        train_batch_size=args.batch_size,
        val_batch_size=val_batch_size,
        val_dataset_size=val_batch_size * args.val_batches,
        num_workers=num_workers,  # This is only applied for train. val has 1 worker
    )
    datamodule.setup()

    # Run the training
    trainer.fit(model, datamodule)
