import argparse
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from constants import LOG_FREQ
from data import CATERDataModule
from data import download_dataset
from model.simone import SIMONE
from utils import wandb_lib
from utils.str2bool import str2bool
from utils.wandb_metrics import WandbCallback


def main():
    args = _parse_args()

    pl.seed_everything(args.seed)

    model = _create_model(args)
    datamodule = _create_datamodule(args)
    logger = _create_logger(args, model)
    trainer = _create_trainer(args, logger)

    trainer.fit(model, datamodule)


def _create_trainer(args: Namespace, pl_logger) -> pl.Trainer:
    wandb_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.environ["EXP_LOG_DIR"], "checkpoints"),
        ),
        WandbCallback(args.gpus * args.batch_size),
    ]
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=wandb_callbacks if args.wandb_logging else None,
        logger=pl_logger,
        precision=16,
        strategy="ddp",
        gradient_clip_val=5,
    )
    return trainer


def _create_logger(args: Namespace, model: pl.LightningModule):
    if args.wandb_logging:
        # Set up wandb logging
        pl_logger = WandbLogger(
            project=args.project,
            name=args.name,
            offline=False,
            log_model=True,  # upload checkpoints to wandb
        )
        # Log gradient and weight histograms
        pl_logger.watch(model, log="all", log_freq=LOG_FREQ * 10)

        # Workaround to fix checkpoint path so wandb uploads them properly.
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/5319#issuecomment-869109468
        if os.environ.get("LOCAL_RANK", None) is None:
            os.environ["EXP_LOG_DIR"] = pl_logger.experiment.dir

        # Something the wandb folks said I needed to include to opt into a new feature that makes lightning + DDP work
        # See https://github.com/wandb/client/issues/2821#issuecomment-949864122
        wandb.require(experiment="service")
    else:
        pl_logger = None

    return pl_logger


def _create_model(args: Namespace) -> SIMONE:
    kwargs = {
        "learning_rate": args.learning_rate,
        "transformer_layers": args.transformer_layers,
        "sigma_x": args.sigma_x,
        "alpha": args.alpha,
        "beta_f": args.beta_f,
        "beta_o": args.beta_o,
    }
    # Load checkpoint if specified
    if args.ckpt_file and args.ckpt_runid:
        checkpoint = wandb_lib.download_file(args.ckpt_runid, args.project, f"checkpoints/{args.ckpt_file}")
        model = SIMONE.load_from_checkpoint(checkpoint, **kwargs)
    else:
        model = SIMONE(**kwargs)
    return model


def _create_datamodule(args: Namespace) -> CATERDataModule:
    download_dataset()
    datamodule = CATERDataModule(
        n_gpus=args.gpus,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        val_dataset_size=args.val_batch_size * args.val_batches,
        num_train_workers=4,
    )
    datamodule.setup()
    return datamodule


def _parse_args() -> Namespace:
    wandb_run_name = f"{datetime.utcnow().isoformat(timespec='seconds')}"

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--wandb_logging", type=str2bool, default=True, help="whether to use wandb logging or not")
    parser.add_argument("--name", type=str, default=wandb_run_name, help="wandb run name (optional)")
    parser.add_argument("--project", type=str, default="simone", help="wandb project name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
        help="memory usage when validating is higher than training; probably need to tune this to your GPU",
    )
    parser.add_argument("--learning_rate", type=float, default=20e-5)
    parser.add_argument("--ckpt_runid", type=str, default=None, help="wandb run to load checkpoint from")
    parser.add_argument("--ckpt_file", type=str, default=None, help="wandb checkpoint filename")
    parser.add_argument("--transformer_layers", type=int, default=4, help="number of layers per transformer stack")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--val_batches", type=int, default=10, help="number of batches to run for validation")

    # loss function
    parser.add_argument("--sigma_x", type=float, default=0.08, help="pixel likelihood prior std")
    parser.add_argument("--alpha", type=float, default=0.2, help="pixel likelihood loss weight")
    parser.add_argument("--beta_o", type=float, default=1e-5, help="object latent loss weight")
    parser.add_argument("--beta_f", type=float, default=1e-4, help="temporal latent loss weight")

    return parser.parse_args()


if __name__ == "__main__":
    main()
