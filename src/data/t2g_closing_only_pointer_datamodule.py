import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.models.components.masking import constants as mc
from src.models.components.masking import utils as masking_utils
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_file,
        dev_file,
        dev_graph_file,
        test_file,
        test_graph_file,
        tokenizer_path,
        maskrules,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
        batch_size: int = 64,
        eval_batch_size: int = 64,
        test_batch_size: int = None,
        use_grouped_sampler: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        pad_to_multiple_of=None,
        var_format="indicator",
        dev_url_patch=None,
        test_url_patch=None,
        add_bos_for_target=True,
        relative_distance=False,
        **kwargs,
    ):
        super().__init__()

        num_workers = min(num_workers, os.cpu_count())
        self.save_hyperparameters(logger=False)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.maskagent = masking_utils.MaskAgent(self.tokenizer, **maskrules)

        assert self.maskagent.maskrules_name in ("closing_only_pointer",)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):

        if self.data_train is not None:
            return

        data_train = self.read_file(self.hparams.train_file)
        data_val = self.read_file(self.hparams.dev_file, self.hparams.dev_url_patch, self.hparams.dev_graph_file)
        data_test = self.read_file(self.hparams.test_file, self.hparams.test_url_patch, self.hparams.test_graph_file)

        def filter_data(data, name):
            _num_orig = len(data)
            data = [
                inst
                for inst in data
                if len(inst["src"]) <= self.hparams.max_src_len and len(inst["tgt"]) <= self.hparams.max_tgt_len
            ]
            if (d := _num_orig - len(data)) > 0:
                log.warning(f"Dropping {d} samples in {name}.")
            log.info(f"There are {len(data)} samples in {name}.")
            return data

        data_train = filter_data(data_train, "TrainSet")
        # data_val = filter_data(data_val, "ValSet")

        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test

    def read_file(self, main_file, url_patch=None, raw_graphs=None, data_root=None):
        data_root = self.hparams.data_root if data_root is None else data_root
        f = open(os.path.join(data_root, main_file))
        f_url = open(os.path.join(data_root, url_patch)) if url_patch is not None else iter(lambda: None, "+")
        f_graph = open(os.path.join(data_root, raw_graphs)) if raw_graphs is not None else iter(lambda: None, "+")

        data = []
        for i, (line, url, graph) in enumerate(zip(f, f_url, f_graph)):
            sids, tids, variables = line.split("\t", 2)
            sids = list(map(int, sids.split(",")))
            tids = list(map(int, tids.split(",")))
            variables = json.loads(variables)
            inst = {"id": i, "src": sids, "tgt": tids, "var": variables}
            if url is not None:
                inst["url"] = url.strip()
            if graph is not None:
                inst["graph"] = graph.strip()
            data.append(inst)

        if raw_graphs is not None:
            f_graph.close()
        if url_patch is not None:
            f_url.close()
        f.close()
        return data

    def get_sampler(self, name):
        if self.hparams.use_grouped_sampler and name == "train":
            return {
                "batch_size": self.hparams.batch_size,
                "sampler": LengthGroupedSampler(
                    batch_size=self.hparams.batch_size,
                    dataset=self.data_train,
                    lengths=[len(item["src"]) for item in self.data_train],
                ),
            }
        if name == "train":
            batch_size = self.hparams.batch_size
        elif name == "val":
            batch_size = self.hparams.eval_batch_size
        elif name == "test":
            batch_size = (
                self.hparams.test_batch_size
                if self.hparams.test_batch_size is not None
                else self.hparams.eval_batch_size
            )
        else:
            raise ValueError
        return {
            "batch_size": batch_size,
            "shuffle": name == "train",
        }

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            **self.get_sampler("train"),
        )
        log.info(f"Train dataloader: {len(loader)}")
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.data_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            **self.get_sampler("val"),
        )
        log.info(f"Val dataloader: {len(loader)}")
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            dataset=self.data_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            **self.get_sampler("test"),
        )
        log.info(f"Test dataloader: {len(loader)}")
        return loader

    def collator(self, data):
        src = [inst["src"] for inst in data]
        tgt = [inst["tgt"] for inst in data]

        if not self.hparams.add_bos_for_target:
            tgt = [item[1:] for item in tgt]

        processed_data = []
        for tgt_item in tgt:
            chunk = self.maskagent(np.array([self.tokenizer.eos_token_id] + tgt_item[:-1]), np.array(tgt_item))

            try:
                _len = chunk.labels.tolist().index(self.tokenizer.eos_token_id) + 1
            except ValueError:  # more than 1 chunk
                _len = len(chunk.labels)

            _offset = chunk.attn_relpos.shape[1] // 2
            processed_data.append(
                (
                    chunk.inputs[:_len],
                    None,  # chunk.inputs_ttypes[:_len],
                    chunk.labels[:_len],
                    None,  # chunk.labels_ttypes[:_len],
                    chunk.attn_mask[:_len, :_len],
                    chunk.attn_relpos[:_len, _offset : _offset + _len],
                    chunk.depth[:_len],
                    chunk.closing_pointers[:_len],
                    chunk.closing_pointer_mask[:_len],
                )
            )

        batch = self.tokenizer.pad(
            {"input_ids": src},
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
        )
        batch["id"] = [item["id"] for item in data]
        batch["decoder_input_ids"] = pad(
            [item[0] for item in processed_data],
            self.tokenizer.pad_token_id,
            self.hparams.pad_to_multiple_of,
        )
        # batch["input_ttypes"] = pad(
        #     [item[1] for item in processed_data], mc.PAD, self.hparams.pad_to_multiple_of
        # )
        batch["labels"] = pad([item[2] for item in processed_data], -100, self.hparams.pad_to_multiple_of)
        # batch["labels_ttypes"] = pad(
        #     [item[3] for item in processed_data], mc.PAD, self.hparams.pad_to_multiple_of
        # )
        batch["decoder_attention_mask"] = pad([item[4] for item in processed_data], 0, self.hparams.pad_to_multiple_of)
        batch["closing_pointers"] = pad([item[7] for item in processed_data], -100, self.hparams.pad_to_multiple_of)
        batch["closing_pointer_mask"] = pad([item[8] for item in processed_data], 0, self.hparams.pad_to_multiple_of)

        if self.hparams.relative_distance:
            batch["relative_distance"] = pad([item[5] for item in processed_data], 0, self.hparams.pad_to_multiple_of)

        if "url" in data[0]:
            batch["url"] = [item["url"] for item in data]
        if "graph" in data[0]:
            batch["graph"] = [item["graph"] for item in data]

        max_tgt_len = batch["decoder_input_ids"].shape[1]
        if self.hparams.var_format == "indicator":
            batch |= self.process_var_indicator(data, processed_data, max_tgt_len)
        elif self.hparams.var_format == "id":
            batch |= self.process_var_id(data, processed_data, max_tgt_len)
        elif self.hparams.var_format == "target-side-pointer":
            batch |= self.process_var_tgt_side_pointer(data, processed_data, max_tgt_len, null_to_self=False)
        elif self.hparams.var_format == "target-side-pointer2":
            batch |= self.process_var_tgt_side_pointer(data, processed_data, max_tgt_len, null_to_self=True)

        return batch

    def process_var_indicator(self, data, processed_data, max_tgt_len):
        # mark positions that should have different alignment.
        # This is only for pt because we use it too group reentrancies.
        neq_ind = torch.zeros(len(data), max_tgt_len, max_tgt_len, dtype=torch.float)
        eq_ind = torch.zeros(len(data), max_tgt_len, max_tgt_len, dtype=torch.float)
        for i, inst in enumerate(data):
            offset = self._get_var_offset(inst, processed_data[i])
            groups = [item for subgroups in inst["var"].values() for item in subgroups.values()]
            for gi1, g1 in enumerate(groups):
                for gi2, g2 in enumerate(groups):
                    if gi1 != gi2:
                        for t1 in g1:
                            for t2 in g2:
                                neq_ind[i, t1 + offset[t1], t2 + offset[t2]] = 1
                for ti1, t1 in enumerate(g1):
                    for ti2, t2 in enumerate(g1):
                        if ti1 != ti2:
                            eq_ind[i, t1 + offset[t1], t2 + offset[t2]] = 1

        return {"tgt_pt_neq_ind": neq_ind, "tgt_pt_eq_ind": eq_ind}

    def process_var_id(self, data, processed_data, max_tgt_len):
        ids = torch.zeros(len(data), max_tgt_len, dtype=torch.long)
        for i, inst in enumerate(data):
            groups = [item for subgroups in inst["var"].values() for item in subgroups.values()]
            offset = self._get_var_offset(inst, processed_data[i])
            for gi, group in enumerate(groups, start=1):  # 0 collects rests
                for t in group:
                    ids[i, t + offset[t]] = gi

        return {"tgt_pt_id": ids}

    def process_var_tgt_side_pointer(self, data, processed_data, max_tgt_len, null_to_self=False):
        ids = torch.full((len(data), max_tgt_len), -100, dtype=torch.long)
        for i, inst in enumerate(data):
            groups = [item for subgroups in inst["var"].values() for item in subgroups.values()]
            offset = self._get_var_offset(inst, processed_data[i])
            for gi, group in enumerate(groups, start=1):
                t0 = group[0]
                t0 = t0 + offset[t0]
                ids[i, t0] = t0 + 1 if null_to_self else 0
                for ti, t in enumerate(group[1:], start=1):
                    prev = group[ti - 1]
                    ids[i, t + offset[t]] = prev + offset[prev] + 1

        return {"tgt_pt_pointer": ids}

    def _get_var_offset(self, data_item, processed_data_item):
        if self.maskagent.is_double_closing:
            offset, cnt, prev_is_closing = [], 0, False
            for ttype in processed_data_item[1]:
                if prev_is_closing:
                    prev_is_closing = False
                else:
                    offset.append(cnt)
                    if ttype == mc.CLOSING_NT:
                        cnt += 1
                        prev_is_closing = True
        else:
            offset = [0] * len(data_item["tgt"])
        return offset

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        excluded = []
        if "id" in batch:
            excluded.append(("id", batch.pop("id")))
        if "url" in batch:
            excluded.append(("url", batch.pop("url")))
        if "graph" in batch:
            excluded.append(("graph", batch.pop("graph")))
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in excluded:
            batch[key] = value
        return batch


def pad(arrays, padding_val, pad_to_multiple_of=None):
    max_len = max(item.shape[-1] for item in arrays)

    if pad_to_multiple_of is not None:
        max_len = (max_len + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

    if isinstance(arrays[0], torch.Tensor):

        def transform(x):
            return x

    elif isinstance(arrays[0], np.ndarray):
        transform = torch.from_numpy
    else:
        transform = torch.tensor

    if arrays[0].ndim == 1:
        output = torch.full((len(arrays), max_len), padding_val, dtype=torch.long)
        for i, arr in enumerate(arrays):
            output[i, : len(arr)] = transform(arr)
    elif arrays[0].ndim == 2:
        output = torch.full((len(arrays), max_len, max_len), padding_val, dtype=torch.long)
        for i, arr in enumerate(arrays):
            output[i, : len(arr), : len(arr)] = transform(arr)
    elif arrays[0].ndim == 3:
        output = torch.full((len(arrays), arrays[0].shape[0], max_len, max_len), padding_val, dtype=torch.long)
        for i, arr in enumerate(arrays):
            _size = arr.shape[1]
            output[i, :, :_size, :_size] = transform(arr)
    else:
        raise ValueError
    return output


if __name__ == "__main__":

    datamodule = Text2GraphDataModule(
        "data/AMR/bart",
        "amrtoken/train.tsv",
        "amrtoken/dev.tsv",
        "amrtoken/dev.graph",
        "amrtoken/test.tsv",
        "amrtoken/test.graph",
        tokenizer_path="data/AMR/bart/model-base-amrtoken",
        maskrules=dict(name="closing_only_pointer"),
        batch_size=4,
    )
    datamodule.setup()
    tokenizer = datamodule.tokenizer
    print("Loaded.")

    print("=" * 80)
    for i, batch in enumerate(datamodule.test_dataloader()):
        if i < 5:
            print(batch["id"], len(batch["input_ids"][0]))
            print(datamodule.tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
            print(datamodule.tokenizer.convert_ids_to_tokens(batch["decoder_input_ids"][0]))
            print(datamodule.tokenizer.convert_ids_to_tokens(batch["labels"][0].clamp(0)))
            print(batch["input_ids"][0][:10])
            print(batch["decoder_input_ids"][0][:10])
            print(batch["closing_pointers"][0][:10])
            print(batch["labels"][0][:10])

            print("=" * 80)
            # breakpoint()
        else:
            break
