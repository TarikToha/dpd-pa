import os
import time

import pytorch_lightning as pl
import torch.backends.cudnn
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger

from src.data import DPDDataModule
from src.model import DPDModelModule

pl.seed_everything(0)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(TQDMProgressBar, "progress_bar")

        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir=f'{int(time.time())}'
                ),
                "model_checkpoint.monitor": "val_evm",
                "model_checkpoint.mode": "min",
                "model_checkpoint.filename": "best-step-{step}-{val_evm:.4f}",
                "model_checkpoint.save_last": True,

                "progress_bar.refresh_rate": 1000,
            }
        )


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = MyLightningCLI(
    DPDModelModule,
    DPDDataModule,
    save_config_kwargs={"overwrite": True},
)

# Copy the config into the experiment directory
# Fix for https://github.com/Lightning-AI/lightning/issues/17168
try:
    os.rename(
        os.path.join(cli.trainer.logger.save_dir, "config.yaml"),  # type:ignore
        os.path.join(
            cli.trainer.checkpoint_callback.dirpath[:-12], "config.yaml"  # type:ignore
        ),
    )
except:
    pass
