import logging
import os
import re
import time
from typing import Sequence

import psutil
import torch.distributed as dist
import torchmetrics
from amrlib.evaluate.smatch_enhanced import (
    compute_scores,
    compute_smatch,
    get_entries,
    redirect_smatch_errors,
)

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

redirect_smatch_errors("smatch.err")


class SmatchMetric:
    def __init__(self) -> None:
        self.preds = []
        self.golds = []
        self.ids = []

    def __call__(self, ids, preds, golds, urls):
        assert len(preds) == len(golds)
        if urls is not None:
            assert len(preds) == len(urls)

            for i in range(len(preds)):
                preds[i] = preds[i].replace("http://link", urls[i])
        self.ids.extend(ids)
        self.preds.extend(self.clean(preds))
        self.golds.extend(golds)

    def compute(self):
        ws = dist.get_world_size() if dist.is_initialized() else 1

        if ws > 1:
            rank = dist.get_rank()
            _buffer = [None] * ws
            dist.all_gather_object(_buffer, (self.ids, self.preds, self.golds))

            if rank == 0:
                all_preds, all_golds, added_ids = [], [], set()
                for ids, preds, golds in _buffer:
                    for id_, pred, gold in zip(ids, preds, golds):
                        if id_ in added_ids:
                            continue
                        all_preds.append(pred)
                        all_golds.append(gold)
                        added_ids.add(id_)
                result = self._compute(all_preds, all_golds)
                logging.info(f"Compute Smatch on {len(added_ids)} samples.")
                _smatch_sync = [result] * ws
                dist.broadcast_object_list(_smatch_sync)
            else:
                _smatch_sync = [None] * ws
                dist.broadcast_object_list(_smatch_sync)
                result = _smatch_sync[rank]
        else:
            result = self._compute(self.preds, self.golds)

        self.preds.clear()
        self.golds.clear()
        return result

    def _compute(self, preds, golds):
        processes = 8
        while processes > 0:
            try:
                precision, recall, f_score = compute_smatch(preds, golds, processes)
            except OSError:
                print(r"RAM memory % used:", psutil.virtual_memory()[2])
                print(r"RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
                processes //= 2
            else:
                break
        return {"P": precision, "R": recall, "F": f_score}

    def compute_all(self):
        compute_scores(self.golds, self.preds)
        return self.compute()

    def clean(self, raw_entries):
        # based on amrlib.evaluate.smatch_enhanced.get_entries
        entries = []
        for e in raw_entries:
            lines = [line.strip() for line in e.splitlines()]
            lines = [line for line in lines if (line and not line.startswith("#"))]
            string = " ".join(lines)
            string = string.replace("\t", " ")  # replace tabs with a space
            string = re.sub(" +", " ", string)  # squeeze multiple spaces into a single
            if string:
                entries.append(string)
        return entries


class SacreBLEUScore(torchmetrics.SacreBLEUScore):
    def __init__(self, lang="en") -> None:
        super().__init__(tokenize="zh" if lang == "zh" else "13a", lowercase=True)
        self.lang = lang

    def update(self, preds: Sequence[str], target: Sequence[str], urls) -> None:
        assert len(preds) == len(target)
        if urls is not None:
            assert len(preds) == len(urls)

            for i in range(len(preds)):
                preds[i] = preds[i].replace("http://link", urls[i])

        return super().update(preds, [[item] for item in target])

    def compute(self):
        return super().compute().item()

