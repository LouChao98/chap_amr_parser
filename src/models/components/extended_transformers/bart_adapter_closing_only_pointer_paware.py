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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation.beam_search import BeamScorer
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    BeamSearchOutput,
    GreedySearchOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    CrossEntropyLoss,
    Seq2SeqLMOutput,
    shift_tokens_right,
)
from transformers.utils import logging

from src.models.components.extended_transformers.bart_adapter_closing_only_pointer import (
    BartForConditionalGeneration as OrigBartForConditionalGeneration,
)
from src.models.components.extended_transformers.bart_paware_strict import (
    BartForConditionalGenerationAndPointer as PAware,
)
from src.models.components.extended_transformers.constrained_decoding import (
    TransformerGrammarClosingOnlyPointerMaskRules,
)

from .bart_paware_strict import PAwareBeamSearchOutput, PAwareGreedyOutput

logger = logging.get_logger(__name__)


class BartForConditionalGeneration(OrigBartForConditionalGeneration):
    default_pointer_args = PAware.default_pointer_args

    def __init__(self, config: BartConfig, maskagent, tg_args, pointer_args):
        super().__init__(config, maskagent, tg_args, pointer_args)

        config = self.config
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

        if config.tg_args["tg_zero_init_up_proj"]:
            for name, param in self.get_decoder().named_parameters():
                if "up_proj" in name or "tg_gating" in name:
                    param.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        relative_distance: Optional[torch.Tensor] = None,
        closing_pointers: Optional[torch.Tensor] = None,
        closing_pointer_mask: Optional[torch.Tensor] = None,
        pointer_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if decoder_inputs_embeds is None:
            decoder = self.get_decoder()
            decoder_inputs_embeds = decoder.embed_tokens(decoder_input_ids) * decoder.embed_scale
            decoder_input_ids = None

        if self.paware_mode != "disable":
            clamped_pointer_ids = pointer_ids.clamp(0).unsqueeze(-1).expand(-1, -1, decoder_inputs_embeds.shape[-1])
            if self.config.pointer_args["pointer_use_pos_emb_only"]:
                # if self.pointer_net.is_generating and self.pointer_net.generation_hidden_states_cache_l0 is not None:
                #     _inp = self.pointer_net.generation_hidden_states_cache_l0[:, :, -1]
                # else:
                #     _inp = decoder_inputs_embeds[:, :, -1]
                # pointer_inp_emb = decoder.embed_positions(_inp, 0).to(_inp.device)
                # tgt_pointed ref = pointer_inp_emb.gather(1, clamped_pointer_ids)
                clamped_pointer_ids = pointer_ids.clamp(0)
                if past_key_values is not None:
                    offset = past_key_values[0][0][0].shape[2]
                else:
                    offset = 0
                tgt_pointed = super(type(decoder.embed_positions), decoder.embed_positions).forward(
                    clamped_pointer_ids + 2 + offset
                )
                # assert torch.allclose(ref, tgt_pointed)
            else:
                if self.pointer_net.is_generating and self.pointer_net.generation_hidden_states_cache_l0 is not None:
                    tgt_pointed = self.pointer_net.generation_hidden_states_cache_l0.gather(1, clamped_pointer_ids)
                else:
                    device = decoder_inputs_embeds.device
                    positions = decoder.embed_positions(decoder_inputs_embeds[:, :, -1], 0).to(device)
                    pointer_inp_emb = decoder.layernorm_embedding(decoder_inputs_embeds + positions)
                    tgt_pointed = pointer_inp_emb.gather(1, clamped_pointer_ids)
                    tgt_pointed = torch.nn.functional.dropout(
                        tgt_pointed, p=self.config.dropout, training=self.training
                    )
            if self.paware_mode == "enable_pre_mask":
                tgt_pointed[pointer_ids == -100] = 0
            residual = tgt_pointed
            tgt_pointed = self.input_reencoding(tgt_pointed)
            if self.config.pointer_args["pointer_adapter_residual"]:
                tgt_pointed = residual + tgt_pointed
            tgt_pointed = self.pointer_adapter_layernorm(tgt_pointed)
            if self.paware_mode == "enable_post_mask":
                tgt_pointed[pointer_ids == -100] = 0
            decoder_inputs_embeds = decoder_inputs_embeds + tgt_pointed

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            relative_distance=relative_distance,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        predicted_pointers = self.pointer_net(outputs, closing_pointer_mask)
        pointer_loss = None
        if closing_pointers is not None and not self.pointer_net.is_generating:
            pointer_loss = self.pointer_loss_fn(predicted_pointers.transpose(1, 2), closing_pointers)
        assert (masked_lm_loss is None) == (pointer_loss is None)

        if not return_dict:
            raise NotImplementedError
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        output = Seq2SeqLMOutput(
            loss=(masked_lm_loss + pointer_loss * self.pointer_loss_strength) if masked_lm_loss is not None else None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        output.pointers = predicted_pointers
        output.logs = {"ml_loss": masked_lm_loss, "match_loss": pointer_loss}
        return output

    def generate(self, **kwargs):
        return PAware.generate(self, **kwargs)

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        pointer_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        pointer_logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # pointer_ids.zero_()

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        pointer_logits_processor = (
            pointer_logits_processor if pointer_logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, pointer_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # print(self.pointer_net.generation_hidden_states_cache_h1.flatten(1).sum(1))
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_pointer_logits = outputs.pointers[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_pointer_logits = pointer_logits_processor(pointer_ids, next_pointer_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # only before normal tokens
            is_normal = self.normal_tokens.gather(0, input_ids.flatten()).view(input_ids.shape)
            is_normal[:, :-1] = is_normal[:, 1:].clone()
            tgt_is_pointable = is_normal

            # no duplicated pointer
            should_mask = (
                (input_ids[:, -1] == self.maskagent.ranges.closing_non_terminal)
                & (next_tokens == self.maskagent.ranges.closing_non_terminal)
            ).unsqueeze(1)
            patch = torch.nn.functional.one_hot(pointer_ids[:, -1].clamp(0), num_classes=tgt_is_pointable.shape[1]).to(
                torch.bool
            )
            patch = should_mask * patch
            tgt_is_pointable = tgt_is_pointable & ~patch

            if tgt_is_pointable.shape[1] > 1:
                tgt_is_pointable[:, 1] = 1

            next_pointer_logits[~tgt_is_pointable] = float("-inf")
            next_pointers = torch.argmax(next_pointer_logits, dim=-1)

            next_pointers[next_tokens != self.maskagent.ranges.closing_non_terminal] = -100

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            pointer_ids = torch.cat([pointer_ids, next_pointers[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        output = PAwareGreedyOutput(
            sequences=input_ids,
            pointers=pointer_ids,
            scores=scores,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
        return output

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        pointer_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        pointer_logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        pruning_token=0,  # 0 means no pruning
        pruning_pointer=0,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # pointer_ids.zero_()

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        pointer_logits_processor = (
            pointer_logits_processor if pointer_logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, pointer_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # print(self.pointer_net.generation_hidden_states_cache_h1.flatten(1).sum(1))
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # (batch_size * num_beams, vocab_size)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None]

            next_pointer_logits = outputs.pointers[:, -1, :]
            next_pointer_logits = pointer_logits_processor(pointer_ids, next_pointer_logits)

            # only before normal tokens
            is_normal = self.normal_tokens.gather(0, input_ids.flatten()).view(input_ids.shape)
            is_normal[:, :-1] = is_normal[:, 1:].clone()
            tgt_is_pointable = is_normal

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if (pt := pruning_token) > 0:
                next_token_scores, selected_tokens = torch.topk(
                    next_token_scores, pt, dim=1, largest=True, sorted=False
                )
            else:
                pt = next_token_scores.shape[1]
                selected_tokens = torch.arange(pt, device=next_token_scores.device)
                selected_tokens = selected_tokens.expand_as(next_token_scores)

            # no duplicated pointer
            should_mask = (
                (input_ids[:, -1, None].repeat(1, pt) == self.maskagent.ranges.closing_non_terminal)
                & (selected_tokens == self.maskagent.ranges.closing_non_terminal)
            ).unsqueeze(1)
            patch = torch.nn.functional.one_hot(pointer_ids[:, -1].clamp(0), num_classes=tgt_is_pointable.shape[1]).to(
                torch.bool
            )
            patch = should_mask * patch.unsqueeze(2)
            tgt_is_pointable = tgt_is_pointable.unsqueeze(-1)
            tgt_is_pointable = tgt_is_pointable & ~patch
            if tgt_is_pointable.shape[1] > 1:
                tgt_is_pointable[:, 1] = 1

            next_pointer_logits = next_pointer_logits.unsqueeze(-1).repeat(1, 1, pt)
            next_pointer_logits[~tgt_is_pointable] = torch.finfo(next_pointer_logits.dtype).min

            can_point_here = selected_tokens == self.maskagent.ranges.closing_non_terminal
            next_pointer_logits.masked_fill_(~can_point_here.unsqueeze(1), torch.finfo(next_pointer_logits.dtype).min)
            if next_pointer_logits.shape[1] == 1:
                next_pointer_logits[:, 0].masked_fill_(~can_point_here, 0.0)
            else:
                next_pointer_logits[:, 1].masked_fill_(~can_point_here, 0.0)

            next_pointer_scores = next_pointer_logits.clamp(torch.finfo(next_pointer_logits.dtype).min).log_softmax(1)

            if (pp := pruning_pointer) > 0 and pp < next_pointer_scores.shape[1]:
                next_pointer_scores, selected_pointers = torch.topk(
                    next_pointer_scores, pp, dim=1, largest=True, sorted=False
                )
            else:
                pp = next_pointer_scores.shape[1]
                selected_pointers = torch.arange(pp, device=next_pointer_scores.device)
                selected_pointers = selected_pointers.unsqueeze(-1)
                selected_pointers = selected_pointers.expand_as(next_pointer_scores)
                selected_pointers = selected_pointers.contiguous()

            next_token_scores = next_token_scores.unsqueeze(1) + next_pointer_scores
            next_token_scores = next_token_scores.clamp(torch.finfo(next_pointer_scores.dtype).min)
            next_token_scores = next_token_scores.view(batch_size, num_beams * pp * pt)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, pp * pt, rounding_mode="floor")
            next_token_pointers = next_tokens % (pp * pt)

            next_tokens = next_token_pointers % pt
            next_pointers = torch.div(next_token_pointers, pt, rounding_mode="floor")
            next_pointers = (
                selected_pointers.view(batch_size, num_beams, pp, pt)
                .gather(1, next_indices[..., None, None].expand(-1, -1, pp, pt))
                .gather(3, next_tokens[..., None, None].expand(-1, -1, pp, -1))
                .squeeze(3)
                .gather(2, next_pointers.unsqueeze(-1))
                .squeeze(2)
            )
            next_tokens = (
                selected_tokens.view(batch_size, num_beams, -1)
                .gather(1, next_indices.unsqueeze(-1).expand(-1, -1, pt))
                .gather(2, next_tokens.unsqueeze(-1))
                .squeeze(-1)
            )

            next_pointers[next_tokens != self.maskagent.ranges.closing_non_terminal] = -100

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                pointer_ids,
                next_token_scores,
                next_tokens,
                next_pointers,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_next_pointers = beam_outputs["next_beam_pointers"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            pointer_ids = torch.cat([pointer_ids[beam_idx, :], beam_next_pointers.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices)))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            pointer_ids,
            beam_scores,
            next_tokens,
            next_pointers,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            return PAwareBeamSearchOutput(
                sequences=sequence_outputs["sequences"],
                pointers=sequence_outputs["pointers"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )

        else:
            return (sequence_outputs["sequences"], sequence_outputs["pointers"])

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
            self.py_mask_rules = TransformerGrammarClosingOnlyPointerMaskRules(
                len(decoder_input_ids),
                self.maskagent.ranges,
                compute_relative_distance=self.config.tg_args["use_relative_positional_encoding"],
            )

        if self.config.tg_args["use_relative_positional_encoding"]:
            att_mask, pointer_mask, relative_distance = self.py_mask_rules.step(
                decoder_input_ids[:, -1], pointer_ids[:, -1]
            )
            relative_distance = torch.from_numpy(relative_distance).to(decoder_input_ids.device)
        else:
            att_mask, pointer_mask = self.py_mask_rules.step(decoder_input_ids[:, -1], pointer_ids[:, -1])
            relative_distance = None
        decoder_attention_mask = torch.from_numpy(att_mask).to(decoder_input_ids.device)
        pointer_mask = torch.from_numpy(pointer_mask).to(decoder_input_ids.device)

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            pointer_ids = pointer_ids[:, -1:]

        # pointer_ids[pointer_ids == 0] = -100

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "relative_distance": relative_distance,
            "pointer_ids": pointer_ids,
            "closing_pointer_mask": pointer_mask == 0,
        }

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

        return reordered_past

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        # do not re-init missing parameters. we have inited them in __init__
        # this may effect unexpected codes, but I did not find any usage except for the initialization
        return []

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        pointer_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if pointer_ids is not None:
            pointer_ids = pointer_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("token_type_ids") is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get("encoder_outputs")
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                expand_size, dim=0
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
            if decoder_attention_mask is not None:
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.repeat_interleave(expand_size, dim=0)

        return input_ids, pointer_ids, model_kwargs
