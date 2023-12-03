import io
import os
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import (
    CustomProgress,
    MetricsTextColumn,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from rich import get_console, reconfigure
from tqdm import tqdm

import wandb
from src.utils.log_utils import rich_theme
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CustomProgressBar(TQDMProgressBar):
    """Only one, short, ascii."""

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Validation sanity check",
            position=self.process_position,
            disable=self.is_disabled,
            leave=False,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=self.process_position,
            disable=self.is_disabled,
            leave=True,
            smoothing=0,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Testing",
            position=self.process_position,
            disable=self.is_disabled,
            leave=True,
            smoothing=0,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"[{trainer.current_epoch + 1}] train")

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"[{trainer.current_epoch + 1}] val")

    def print(
        self,
        *args,
        sep: str = " ",
        end: str = os.linesep,
        file: Optional[io.TextIOBase] = None,
        nolock: bool = False,
    ):
        log.info(sep.join(map(str, args)))
        # active_progress_bar = None
        #
        # if self.main_progress_bar is not None and not self.main_progress_bar.disable:
        #     active_progress_bar = self.main_progress_bar
        # elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
        #     active_progress_bar = self.val_progress_bar
        # elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
        #     active_progress_bar = self.test_progress_bar
        # elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
        #     active_progress_bar = self.predict_progress_bar
        #
        # if active_progress_bar is not None:
        #     s = sep.join(map(str, args))
        #     active_progress_bar.write(s, end=end, file=file, nolock=nolock)


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("console_kwargs", {"theme": rich_theme})
        super().__init__(*args, **kwargs)

    def print(
        self,
        *args,
        sep: str = " ",
        end: str = os.linesep,
        file: Optional[io.TextIOBase] = None,
        nolock: bool = False,
    ):
        log.info(sep.join(map(str, args)))

    def _get_train_description(self, current_epoch: int) -> str:

        train_description = f"Epoch {current_epoch}"
        # patch for max_epochs=-1
        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 0:
            train_description += f"/{self.trainer.max_epochs - 1}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()

            if os.environ.get("SLURM_JOB_ID") is not None:
                self._console.width = 140

            self._metric_component = MetricsTextColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False


class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("tags") is not None:
            kwargs["tags"] = [item for item in kwargs["tags"] if item is not None]
        else:
            kwargs["tags"] = []
        kwargs["tags"] += self.collect_hydra_group_choices()
        super().__init__(*args, **kwargs)

    def finalize(self, status: str) -> None:
        if not dist.is_initialized() or dist.get_rank() == 0:
            for fname in Path(self.save_dir).glob("*"):
                _fname = str(fname)
                if "pred_test" in _fname or "pred_val" in _fname or "train.log" in _fname or "test.log" in _fname:
                    wandb.save(_fname, base_path=str(fname.parent))
        return super().finalize(status)

    def collect_hydra_group_choices(self):
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_config = HydraConfig.get()
            choices = hydra_config.runtime.choices
            tags = []
            for name, choice in choices.items():
                if (
                    name
                    not in (
                        "local",
                        "hparams_search",
                        "hydra",
                        "extras",
                        "paths",
                        "trainer",
                        "logger",
                        "callbacks",
                    )
                    and not name.startswith("hydra/")
                    and choice is not None
                ):
                    tags.append(f"{name}={choice}")
            return tags
        except ValueError:
            return []
