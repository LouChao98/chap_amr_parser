import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.models.components.masking import utils as masking_utils
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphForPredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        source_file,
        tokenizer_path,
        maskrules,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        pad_to_multiple_of=None,
        **kwargs,
    ):
        super().__init__()

        num_workers = min(num_workers, os.cpu_count())
        self.save_hyperparameters(logger=False)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.maskagent = masking_utils.MaskAgent(self.tokenizer, **maskrules)

        self.data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data is not None:
            return

        data = self.read_file(self.hparams.source_file)
        self.data = data

    def read_file(self, main_file, data_root=None):
        data_root = self.hparams.data_root if data_root is None else data_root
        f = open(os.path.join(data_root, main_file))

        data = []
        for i, line in enumerate(f):
            inst = {"id": i, "src": line.strip()}
            data.append(inst)

        f.close()
        return data

    def get_sampler(self, name):
        return {
            "batch_size": self.hparams.batch_size,
            "shuffle": name == "train",
        }

    def predict_dataloader(self):
        loader = DataLoader(
            dataset=self.data,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            **self.get_sampler("predict"),
        )
        log.info(f"Predict dataloader: {len(loader)}")
        return loader

    def collator(self, data):
        src = [inst["src"] for inst in data]

        batch = self.tokenizer(
            src,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
        )
        batch["id"] = [item["id"] for item in data]
        return batch

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        excluded = []
        if "id" in batch:
            excluded.append(("id", batch.pop("id")))
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in excluded:
            batch[key] = value
        return batch
