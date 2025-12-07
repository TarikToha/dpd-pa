import glob
import os
from typing import Tuple

import pytorch_lightning as pl
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection, MeanSquaredError
from transformers.optimization import get_cosine_schedule_with_warmup

from src.metrics import NMSE_dB, EVM_rms


class PAModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: [B, T, 2]
        out = self.fc(out)  # [B, T, 2]
        return out


class DPDModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: [B, T, 2]
        out = self.fc(out)  # [B, T, 2]
        return out


class PAModelModule(pl.LightningModule):
    def __init__(
            self,
            optimizer: str = "adamw",
            lr: float = 1e-2,
            betas: Tuple[float, float] = (0.9, 0.999),
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            scheduler: str = "cosine",
            warmup_steps: int = 0,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.hidden_state = None

        # Initialize network with random weights
        self.net = PAModel()

        # Define metrics
        self.train_metrics = MetricCollection({
            "mse": MeanSquaredError(),
        })

        self.val_metrics = MetricCollection({
            "nmse": NMSE_dB(),
            "evm": EVM_rms(),
        })

        # Define loss
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        x, y = batch

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"train_metrics")(pred, y)

        # Log
        self.log(f"train_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"train_{k.lower()}", v, on_epoch=True)

        return loss

    def on_validation_start(self):
        self.hidden_state = None

    def validation_step(self, batch, _):
        x, y = batch

        # Pass through network
        out, self.hidden_state = self.net.lstm(x, self.hidden_state)
        pred = self.net.fc(out)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"val_metrics")(pred, y)

        # Log
        self.log(f"val_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"val_{k.lower()}", v, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )

        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)

        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class DPDModelModule(pl.LightningModule):
    def __init__(
            self,
            optimizer: str = "adamw",
            lr: float = 1e-2,
            betas: Tuple[float, float] = (0.9, 0.999),
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            scheduler: str = "cosine",
            warmup_steps: int = 0,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.hidden_state = None

        # Initialize network
        pa_ckpt_path = glob.glob('pa_model/lightning_logs/version_0/checkpoints/best*')[0]
        assert os.path.exists(pa_ckpt_path), 'pa_model checkpoint not found'

        self.pa_model = PAModelModule.load_from_checkpoint(checkpoint_path=pa_ckpt_path)
        self.pa_model.cuda().eval()
        for param in self.pa_model.parameters():
            param.requires_grad = False

        self.dpd_model = DPDModel()

        # Define metrics
        self.train_metrics = MetricCollection({
            "mse": MeanSquaredError(),
        })

        self.val_metrics = MetricCollection({
            "nmse": NMSE_dB(),
            "evm": EVM_rms(),
        })

        # Define loss
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.dpd_model(x)

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        x, y = batch

        # Pass through network
        x_hat = self.dpd_model(x)
        pred = self.pa_model(x_hat)

        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"train_metrics")(pred, y)

        # Log
        self.log(f"train_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"train_{k.lower()}", v, on_epoch=True)

        return loss

    def on_validation_start(self):
        self.dpd_hidden_state = None
        self.pa_hidden_state = None

    def validation_step(self, batch, _):
        x, y = batch

        # Pass through network
        out, self.dpd_hidden_state = self.dpd_model.lstm(x, self.dpd_hidden_state)
        x_hat = self.dpd_model.fc(out)

        out, self.pa_hidden_state = self.pa_model.net.lstm(x_hat, self.pa_hidden_state)
        pred = self.pa_model.net.fc(out)

        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"val_metrics")(pred, y)

        # Log
        self.log(f"val_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"val_{k.lower()}", v, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.dpd_model.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.dpd_model.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.dpd_model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )

        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)

        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
