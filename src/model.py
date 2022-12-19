import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary

from utils import get_lr_schedule, grad_norm, split_decay_params, weight_norm


class FCN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_model()
        self.decay_groups, self.no_decay_groups = split_decay_params(self)

    def build_model(self):
        """Build model layers."""
        layer_dims = [int(i) for i in self.cfg.layer_dims.split("x")]
        self._layers = []
        for i in range(1, len(layer_dims) - 1):
            if self.cfg.dropout > 0:
                drop = (
                    nn.Dropout(self.cfg.dropout)
                    if i != 1
                    else nn.Dropout(self.cfg.dropout + 0.3)
                )
                self._layers.append(drop)
                self.add_module(f"drop{i}", drop)
            fc = nn.Linear(layer_dims[i - 1], layer_dims[i])
            self._layers.append(fc)
            self.add_module(f"fc{i}", fc)
            relu = nn.ReLU()
            self._layers.append(relu)
            self.add_module(f"relu{i}", relu)
        fc = nn.Linear(layer_dims[-2], layer_dims[-1])
        self._layers.append(fc)
        self.add_module(f"out", fc)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self._layers:
            x = layer(x)
        return x

    def on_train_start(self):
        # Log model size
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        self.log("trainable_params", float(trainable_params), logger=True)
        self.log("total_params", float(total_params), logger=True)
        # Print model summary
        if self.cfg.model_summary:
            summary(self, (1, 28 * 28))

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        pred = output.argmax(dim=1)
        acc = pred.eq(y).float().mean()
        self.log(f"{stage}_loss", loss, logger=True)
        self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="train")
        return {"loss": loss, "train_acc": acc}

    def on_after_backward(self):
        # Log gradient norms
        norm = grad_norm(self, self.decay_groups)
        self.log("grad_norm", norm, logger=True)

    def training_epoch_end(self, outputs):
        loss = torch.stack([i["loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_train_acc": acc, "avg_train_loss": loss}, logger=True)
        # Log weight norms
        norm = weight_norm(self, self.decay_groups)
        self.log("weight_norm", norm, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        loss = torch.stack([i["test_loss"] for i in outputs]).double().mean()
        acc = torch.stack([i["test_acc"] for i in outputs]).double().mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optimizer_groups = [
            {
                "params": [param_dict[pn] for pn in self.decay_groups],
                "weight_decay": self.cfg.wd,
            },
            {
                "params": [param_dict[pn] for pn in self.no_decay_groups],
                "weight_decay": 0.0,
            },
        ]
        if self.cfg.optimizer == "adam":
            optimizer = optim.AdamW(optimizer_groups, lr=self.cfg.lr)
        elif self.cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                optimizer_groups, lr=self.cfg.lr, momentum=self.cfg.momentum
            )
        if self.cfg.lr_scheduler:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, get_lr_schedule(self.cfg.lr_scheduler)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer
