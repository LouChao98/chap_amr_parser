import re
from pathlib import Path
from typing import Any, List

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from transformers import AutoModelForSeq2SeqLM

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphLitModule(LightningModule):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        val_metric,
        test_metric,
        save_prediction_dir,
        test_gen_args=None,
        load_from_checkpoint=None,
        fix_parameters_in_reference_model=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        if test_gen_args is None:
            test_gen_args = {}
        self.save_hyperparameters(logger=False)
        self.save_prediction_dir = Path(save_prediction_dir)

        self.net = None
        self.maskagent = None

        # metric objects for calculating and averaging accuracy across batches
        self.val_metric = None
        self.test_metric = None

        # for tracking best so far validation accuracy
        self.val_best = MaxMetric(sync_on_compute=False)
        
        self.val_outputs = []
        self.test_outputs = []

    def setup(self, stage: str, datamodule=None) -> None:
        if self.net is not None:
            return

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule
        self.maskagent = self.datamodule.maskagent

        self.net = instantiate(
            self.hparams.net,
            tokenizer=self.datamodule.tokenizer,
            maskagent=self.maskagent,
        )
        self.val_metric = instantiate(self.hparams.val_metric)
        self.test_metric = instantiate(self.hparams.test_metric)

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(self.hparams.load_from_checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        if self.hparams.fix_parameters_in_reference_model:
            reference_model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.fix_parameters_in_reference_model)
            fixed_parameters = [item[0] for item in reference_model.named_parameters()]
            for name in fixed_parameters:
                p1 = self.net.get_parameter("model." + name)
                p1.requires_grad_(False)

    def forward(self, batch):
        # return type('DummyResult', (object,), {'loss': torch.tensor(0., requires_grad=True)})
        return self.net(batch)

    def generate(self, batch, **kwargs):
        return self.net.generate(batch, **kwargs)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)

        self.log(
            "train/loss",
            output.loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=len(batch["id"]),
        )

        if hasattr(output, "logs"):
            self.log_dict({"train/" + k: v.mean() for k, v in output.logs.items()})

        return output.loss

    def on_validation_start(self):
        self.val_outputs.clear()

    def validation_step(self, batch: Any, batch_idx: int, **kwargs):
        output = self(batch)
        # kwargs |= self.hparams.test_gen_args
        preds = self.generate(batch, **kwargs)
        self.val_metric(batch["id"], [item["graph"] for item in preds], batch["graph"], batch.get("url"))
        self.val_outputs.append({"preds": [pred_item | {"id": i} for i, pred_item in zip(batch["id"], preds)]})
        return {"loss": output.loss}

    def on_validation_epoch_end(self):
        
        if self.trainer.sanity_checking:
            return

        outputs = self.val_outputs
        preds = sum((item["preds"] for item in outputs), list())
        preds.sort(key=lambda x: x["id"])
        self.save_prediction(self.save_prediction_dir / f"pred_val_{self.global_step}.txt", preds)

        metric = self.val_metric.compute()
        self.val_best(metric["F"])  # update best so far val acc

        self.print("val/epoch", str(self.current_epoch))
        self.print("val/metric", str(metric))
        self.log_dict({"val/" + k: v for k, v in metric.items()})

        # log `val_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/bestF", self.val_best.compute(), prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int, **kwargs):
        kwargs |= self.hparams.test_gen_args
        preds = self.generate(batch, **kwargs)
        self.test_metric(batch["id"], [item["graph"] for item in preds], batch["graph"], batch.get("url"))
        output = {"preds": [pred_item | {"id": i} for i, pred_item in zip(batch["id"], preds)]}
        self.test_outputs.append(output)
        return output
        
    def on_test_epoch_end(self):
        outputs = self.test_outputs
        preds = sum((item["preds"] for item in outputs), list())
        preds.sort(key=lambda x: x["id"])
        self.save_prediction(self.save_prediction_dir / "pred_test.txt", preds)

        metric = self.test_metric.compute()
        self.log_dict({"test/" + k: v for k, v in metric.items()})
        self.print({"test/" + k: v for k, v in metric.items()})

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        preds = self.generate(batch, **self.hparams.test_gen_args)
        return {
            "preds": [pred_item | {"id": i} for i, pred_item in zip(batch["id"], preds)],
        }


    def save_prediction(self, path, preds):
        ws = dist.get_world_size() if dist.is_initialized() else 1
        if ws > 1:
            _buffer = [None] * ws
            dist.all_gather_object(_buffer, preds)

            if dist.get_rank() == 0:
                all_preds, added_ids = [], set()
                for preds in _buffer:
                    for pred in preds:
                        if pred["id"] in added_ids:
                            continue
                        all_preds.append(pred)
                        added_ids.add(pred["id"])
                all_preds.sort(key=lambda x: x["id"])
                self._save_prediction(path, all_preds)
        else:
            self._save_prediction(path, preds)

    def _save_prediction(self, path, preds):
        log.info(f"Writing to {path}")
        with open(path, "w") as f:
            for inst in preds:
                # f.write(f'{inst["id"]}\n')
                f.write(f'# ::raw {inst["raw"]}\n')
                f.write(f'# ::snt {inst["snt"]}\n')
                f.write(inst["graph"])
                f.write("\n\n")

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        optimizer_cfg = self.hparams.optimizer
        if optimizer_cfg.groups is None or len(optimizer_cfg.groups) == 0:
            params = [item for item in self.parameters() if item.requires_grad]
        else:
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                matches = [i for i, g in enumerate(optimizer_cfg.groups) if re.match(g.pattern, name)]
                if len(matches) > 1:
                    log.warning(f"{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}")
                if len(matches) > 0:
                    log.debug(f"{name} match {optimizer_cfg.groups[matches[0]].pattern}.")
                    params[matches[0]].append(p)
                else:
                    log.debug(f"{name} match defaults.")
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    log.warning(f"Nothing matches {optimizer_cfg.groups[i].pattern}")
            params = [{"params": p, **optimizer_cfg.groups[i]} for i, p in enumerate(params) if len(p) > 0]
            params.append({"params": default_group})

        optimizer = instantiate(optimizer_cfg.args, params=params, _convert_="all")

        if (scheduler_cfg := self.hparams.scheduler) is None:
            return optimizer

        scheduler = instantiate(scheduler_cfg.args, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.interval,
                "frequency": scheduler_cfg.frequency,
                "monitor": scheduler_cfg.monitor,
                "strict": True,
            },
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        if "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {}
            for k, v in checkpoint.items():
                if k == "module":
                    for name, tensor in checkpoint["module"].items():
                        name = name.replace("_forward_module.", "")
                        checkpoint["state_dict"][name] = tensor

        return super().on_load_checkpoint(checkpoint)
