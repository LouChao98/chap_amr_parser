import re

import numpy as np
import smatch
import torch
import torch.nn as nn
from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer
from transformers import BartTokenizerFast
from transformers.models.bart import modeling_bart

from src.models.components.extended_transformers import (
    bart_adapter,
    bart_adapter_chain,
    bart_fusing,
    bart_inplace,
)
from src.models.components.masking.utils import MaskAgent
from src.models.components.postprocess import (
    postprocess_inline_double,
    postprocess_inline_single,
)
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphInline(nn.Module):
    supported_models = {
        "bart": modeling_bart.BartForConditionalGeneration,
        "bart_inplace": bart_inplace.BartForConditionalGeneration,
        "bart_adapter": bart_adapter.BartForConditionalGeneration,
        "bart_adapter_chain": bart_adapter_chain.BartForConditionalGeneration,
        "bart_fusing": bart_fusing.BartForConditionalGeneration,
    }

    def __init__(
        self,
        tokenizer,
        maskagent,
        model_name,
        path_to_pretrained,
        model_config=None,
        force_bos_token_id=True,
        convert_var=False,
        fix_decoder_bos=False,
        **tg_args,
    ):
        super().__init__()

        self.tokenizer: BartTokenizerFast = tokenizer
        self.maskagent: MaskAgent = maskagent
        self.is_tg = maskagent.is_tg
        self.force_bos_token_id = force_bos_token_id

        self.convert_var = convert_var
        self.fix_decoder_bos = fix_decoder_bos

        MODEL = self.supported_models[model_name]
        model_config = {} if model_config is None else model_config
        if self.is_tg:
            tg_args = MODEL.default_tg_args | tg_args
            log.info(f"Setup {model_name} with args {tg_args=}, {model_config=}")
            self.model = MODEL.from_pretrained(path_to_pretrained, maskagent, tg_args, **model_config)
        else:
            self.model = MODEL.from_pretrained(path_to_pretrained, **model_config)

    def forward(self, inputs):
        extra_args = {}
        if "relative_distance" in inputs:
            extra_args["relative_distance"] = inputs["relative_distance"]
        if self.fix_decoder_bos:
            inputs["decoder_input_ids"][:, 0] = self.model.config.decoder_start_token_id
        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            labels=inputs["labels"],
            decoder_attention_mask=inputs["decoder_attention_mask"] if self.is_tg else None,
            **extra_args,
        )
        return output

    def generate(self, inputs, **kwargs):
        kwargs.setdefault("output_attentions", True)
        kwargs.setdefault("max_new_tokens", 768)
        kwargs.setdefault("early_stopping", True)
        kwargs.setdefault("num_beams", 1)
        kwargs.setdefault("no_repeat_ngram_size", 0)
        kwargs.setdefault("forced_bos_token_id", self.tokenizer.bos_token_id if self.force_bos_token_id else None)

        output = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs,
        )

        output = self.tokenizer.batch_decode(output)
        if self.convert_var:
            output_ = []
            for item in output:
                item = re.sub(r" <v:(\d+)>", r"_\1", item)
                item = re.sub(r" <inv>", "-of", item)
                output_.append(item)

            output = output_

        pp_func = postprocess_inline_double if self.maskagent.is_double_closing else postprocess_inline_single
        output = list(map(pp_func, output))

        deserialized = list(map(lambda x: PenmanDeSerializer(x).get_graph_string(), output))
        snts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        output = [
            {"raw": o, "snt": s, "graph": g if (g is not None) and (g != "()") else "( b / bad )"}
            for o, s, g in zip(output, snts, deserialized, strict=True)
        ]
        return output
