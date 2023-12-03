# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""PyTorch BART model."""

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers.utils import logging

from src.models.components.extended_transformers.bart_adapter import (
    BartForConditionalGeneration,
)
from src.models.components.extended_transformers.bart_paware_strict import (
    BartForConditionalGenerationAndPointer as PAware,
)
from src.models.components.extended_transformers.constrained_decoding import (
    TransformerGrammarDoubleLogitsProcessor,
    TransformerGrammarDoubleMaskRules,
    TransformerGrammarSingleOpenHeadMaskRules,
    TransformerGrammarSingleComposingOnlyMaskRules,
    TransformerGrammarSingleStackingOnlyMaskRules,
    TransformerGrammarSingleDiffHeadsMaskRules,
    TransformerGrammarSingleLogitsProcessor,
    TransformerGrammarSingleMaskRules,
)

logger = logging.get_logger(__name__)


class BartForConditionalGenerationAndPointer(BartForConditionalGeneration):
    default_pointer_args = PAware.default_pointer_args

    def __init__(self, config: BartConfig, maskagent, tg_args, pointer_args):
        config.pointer_args = self.default_pointer_args | pointer_args
        super().__init__(config, maskagent, tg_args)
        # mode 1: linear - nonlinear
        # mode 2: FNN (large mid)
        # mode 3: FNN (small mid)
        self.pointer_adapter_impl = config.pointer_args["pointer_adapter_impl"]
        if self.pointer_adapter_impl > 0:
            if config.pointer_args["pointer_adapter_layernorm"]:
                self.pointer_adapter_layernorm = nn.LayerNorm(config.d_model)
            else:
                self.pointer_adapter_layernorm = nn.Identity()
        if self.pointer_adapter_impl == 1:
            self.input_reencoding = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                ACT2FN[config.activation_function],
            )
        elif self.pointer_adapter_impl == 2:
            self.input_reencoding = nn.Sequential(
                nn.Linear(config.d_model, config.decoder_ffn_dim),
                ACT2FN[config.activation_function],
                nn.Dropout(config.activation_dropout),
                nn.Linear(config.decoder_ffn_dim, config.d_model),
            )
        elif self.pointer_adapter_impl == 3:
            self.input_reencoding = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                ACT2FN[config.activation_function],
                nn.Dropout(config.activation_dropout),
                nn.Linear(config.d_model // 2, config.d_model),
            )

        self.paware_mode = config.pointer_args["paware_mode"]
        self.pointer_loss_strength = config.pointer_args["pointer_loss_strength"]
        self.post_init()

        if self.config.tg_args["tg_fix_init"]:
            logger.info("Fixing initialization")
            self.input_reencoding.apply(self._init_weights)

        if config.pointer_args["pointer_adapter_last_linear_zero_init"]:
            assert self.pointer_adapter_impl in (2, 3)
            self.input_reencoding[-1].weight.data.zero_()
            self.input_reencoding[-1].bias.data.zero_()

    def forward(self, *args, **kwargs):
        return PAware.forward(self, *args, **kwargs)

    def generate(self, **kwargs):
        if self.maskagent.is_double_closing:
            self.tg_logits_processor = TransformerGrammarDoubleLogitsProcessor(self.maskagent.ranges)
        else:
            self.tg_logits_processor = TransformerGrammarSingleLogitsProcessor(self.maskagent.ranges)
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = []
        kwargs["logits_processor"].append(self.tg_logits_processor)
        return PAware.generate(self, **kwargs)

    def greedy_search(self, *args, **kwargs):
        return PAware.greedy_search(self, *args, **kwargs)

    def beam_search(self, *args, **kwargs):
        return PAware.beam_search(self, *args, **kwargs)

    def beam_sample(self, *args, **kwargs):
        return PAware.beam_sample(self, *args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        pointer_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        if decoder_input_ids.shape[1] == 1:
            if self.maskagent.is_double_closing:
                self.py_mask_rules = TransformerGrammarDoubleMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    mode=2 if self.maskagent.maskrules.use_stacking else 1,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            elif self.maskagent.maskrules_name == "single_closing_nt":
                self.py_mask_rules = TransformerGrammarSingleMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            elif self.maskagent.maskrules_name == "single_closing_nt_composing_only":
                self.py_mask_rules = TransformerGrammarSingleComposingOnlyMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            elif self.maskagent.maskrules_name == "single_closing_nt_stacking_only":
                self.py_mask_rules = TransformerGrammarSingleStackingOnlyMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            elif self.maskagent.maskrules_name == "single_closing_nt_open_head":
                self.py_mask_rules = TransformerGrammarSingleOpenHeadMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            elif self.maskagent.maskrules_name == "single_closing_nt_diff_heads":
                self.py_mask_rules = TransformerGrammarSingleDiffHeadsMaskRules(
                    len(decoder_input_ids),
                    self.maskagent.ranges,
                    compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
                )
            else:
                raise ValueError
        att_mask = self.py_mask_rules.step(decoder_input_ids[:, -1])
        decoder_attention_mask = torch.from_numpy(att_mask).to(decoder_input_ids.device)

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            pointer_ids = pointer_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "pointer_ids": pointer_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(*args, **kwargs):
        return PAware._expand_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()

        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    (
                        past_state[0].index_select(0, beam_idx),
                        past_state[1].index_select(0, beam_idx) if past_state[1] is not None else None,
                    )
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )

        self.pointer_net._reorder_cache(beam_idx)
        beam_idx = beam_idx.tolist()
        self.py_mask_rules.reorder(beam_idx)
        self.tg_logits_processor.reorder(beam_idx)
        return reordered_past

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        # do not re-init missing parameters. we have inited them in __init__
        # this may effect unexpected codes, but I did not find any usage except for the initialization
        return []
