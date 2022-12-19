import argparse

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import MNISTDataModule
from model import FCN


def parser_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--data_dir", type=str, default="data")
    # Model
    parser.add_argument("--layer_dims", type=str)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler", type=str, choices=["inverse", "inverse_sqrt"])
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--momentum", type=float, default=0.0)
    # Experiment
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project_id", type=str, default="pytorch-mnist-dropout")
    parser.add_argument("--wandb", action="store_true")
    # Misc
    parser.add_argument("--model_summary", action="store_true")
    cfg = EasyDict(vars(parser.parse_args()))
    return cfg


def main(cfg):
    # Set seed
    if cfg.seed is not None:
        seed_everything(cfg.seed)
    # Datamodule
    datamodule = MNISTDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    # Model
    model = FCN(cfg)
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    mocel_checkpoint = ModelCheckpoint(
        filename="{epoch}-{avg_train_acc:.4f}-{avg_val_acc:.4f}",
        monitor="avg_val_acc",
        save_top_k=5,
        mode="max",
    )
    # Logger
    wandb_logger = WandbLogger(
        offline=not cfg.wandb, project=cfg.project_id, entity="chrisliu298", config=cfg
    )
    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=[lr_monitor, mocel_checkpoint],
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=wandb_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # Train
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    trainer.validate(
        ckpt_path="best", dataloaders=datamodule.val_dataloader(), verbose=False
    )
    trainer.test(
        ckpt_path="best", dataloaders=datamodule.test_dataloader(), verbose=False
    )
    # Wandb finish
    wandb.finish(quiet=True)


if __name__ == "__main__":
    cfg = parser_args()
    main(cfg)
