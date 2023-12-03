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

import copy
import inspect
import warnings
from collections import UserDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.generation import GenerationConfig
from transformers.generation.beam_search import BeamScorer
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration as OrigBartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import logging

GenerateOutput = Union[GreedySearchOutput, BeamSearchOutput, BeamSampleOutput]

logger = logging.get_logger(__name__)


@dataclass
class PAwareGreedyOutput(GreedySearchEncoderDecoderOutput):
    pointers: torch.LongTensor = None


@dataclass
class PAwareBeamSearchOutput(BeamSearchEncoderDecoderOutput):
    pointers: torch.LongTensor = None


@dataclass
class PAwareBeamSampleOutput(BeamSampleEncoderDecoderOutput):
    pointers: torch.LongTensor = None


class PAwareBeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """Initialize n-best list of hypotheses."""
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.beams)

    def add(
        self,
        hyp: torch.LongTensor,
        hyp_pointer: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
    ):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, hyp_pointer, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted((s, idx) for idx, (s, *_) in enumerate(self.beams))
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """If there are enough hypotheses and that none of the hypotheses being generated can
        become better than the worst one in the heap, then we are done with this sentence."""

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def get_scaled_hypo_type(scale):
    class PAwareBeamHypothesesFixLength(PAwareBeamHypotheses):
        def add(
            self,
            hyp: torch.LongTensor,
            hyp_pointer: torch.LongTensor,
            sum_logprobs: float,
            beam_indices: Optional[torch.LongTensor] = None,
        ):
            """Add a new hypothesis to the list."""
            score = sum_logprobs / ((hyp.shape[-1] + (hyp_pointer != -100).sum() ** scale) ** self.length_penalty)
            if len(self) < self.num_beams or score > self.worst_score:
                self.beams.append((score, hyp, hyp_pointer, beam_indices))
                if len(self) > self.num_beams:
                    sorted_next_scores = sorted((s, idx) for idx, (s, *_) in enumerate(self.beams))
                    del self.beams[sorted_next_scores[0][1]]
                    self.worst_score = sorted_next_scores[1][0]
                else:
                    self.worst_score = min(score, self.worst_score)

    return PAwareBeamHypothesesFixLength


class PAwareBeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        pointer_scale=0,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False

        beam_hyp_type = PAwareBeamHypotheses if pointer_scale == 0 else get_scaled_hypo_type(pointer_scale)
        self._beam_hyps = [
            beam_hyp_type(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        # if not isinstance(num_beams, int) or num_beams <= 1:
        #     raise ValueError(
        #         f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
        #         " one should make use of `greedy_search` instead."
        #     )

        # if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
        #     raise ValueError(
        #         "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
        #         f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
        #     )

        # if "max_length" in kwargs:
        #     warnings.warn(
        #         "Passing `max_length` to BeamSearchScorer is deprecated and has no effect. "
        #         "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
        #         ", or `group_beam_search(...)`."
        #     )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        pointer_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_pointers: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_pointers = torch.full((batch_size, self.group_size), -100, dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_pointer, next_score, next_index) in enumerate(
                zip(
                    next_tokens[batch_idx],
                    next_pointers[batch_idx],
                    next_scores[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        pointer_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_pointers[batch_idx, beam_idx] = next_pointer
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                logger.error(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )
                # raise ValueError(
                #     f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                #     f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                # )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_pointers": next_beam_pointers.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        pointer_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_pointers: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                final_pointers = pointer_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_pointers, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_pointers = best_hyp_tuple[2]
                best_index = best_hyp_tuple[3]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append((best_hyp, best_pointers))

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        decoded_pointers: torch.LongTensor = input_ids.new_full(
            (batch_size * self.num_beam_hyps_to_keep, sent_max_len), -100
        )

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, ((hypo, hypo_pointers), best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo
            decoded_pointers[i, : sent_lengths[i]] = hypo_pointers

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "pointers": decoded_pointers,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BartForConditionalGenerationAndPointer(OrigBartForConditionalGeneration):
    # -+-- adapter --+-- layernorm --
    #  +-------------+
    default_pointer_args = {
        "paware_mode": "enable_post_mask",
        "pointer_loss_strength": 1.0,
        "pointer_adapter_impl": 3,
        "pointer_adapter_layernorm": False,
        "pointer_adapter_last_linear_zero_init": True,
        "pointer_adapter_residual": False,
        "pointer_use_pos_emb_only": False,
    }

    def __init__(self, config: BartConfig, pointer_args):
        self.pointer_net = None
        self.pointer_loss_fn = None
        self.tokenizer = None

        config.pointer_args = self.default_pointer_args | pointer_args
        super().__init__(config)

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

        logger.info("Fixing initialization")
        self.input_reencoding.apply(self._init_weights)

        if config.pointer_args["pointer_adapter_last_linear_zero_init"]:
            assert self.pointer_adapter_impl in (2, 3)
            self.input_reencoding[-1].weight.data.zero_()
            self.input_reencoding[-1].bias.data.zero_()

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
        pointer_ids: Optional[torch.LongTensor] = None,
        pointer_labels: Optional[torch.LongTensor] = None,
        pointer_mask: Optional[torch.LongTensor] = None,
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
                if self.pointer_net.is_generating and self.pointer_net.generation_hidden_states_cache_l0 is not None:
                    _inp = self.pointer_net.generation_hidden_states_cache_l0[:, :, -1]
                else:
                    _inp = decoder_inputs_embeds[:, :, -1]
                pointer_inp_emb = decoder.embed_positions(_inp, 0).to(_inp.device)
                tgt_pointed = pointer_inp_emb.gather(1, clamped_pointer_ids)
                # clamped_pointer_ids = pointer_ids.clamp(0)
                # if past_key_values is not None:
                #     offset = past_key_values[0][0][0].shape[2]
                # else:
                #     offset = 0
                # tgt_pointed = super(type(decoder.embed_positions), decoder.embed_positions).forward(
                #     clamped_pointer_ids + 2 + offset
                # )
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
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        predicted_pointers = self.pointer_net(outputs, pointer_mask)
        pointer_loss = None
        if pointer_labels is not None:
            pointer_loss = self.pointer_loss_fn(predicted_pointers.transpose(1, 2), pointer_labels)
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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = False,
        pointer_logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        with self.pointer_net.generation_context():
            if "beam_search_pointer_scale" in kwargs:
                beam_search_pointer_scale = kwargs.pop("beam_search_pointer_scale")
            else:
                beam_search_pointer_scale = 0

            if "consider_pointer_prob_in_topk" in kwargs:
                consider_pointer_prob_in_topk = kwargs.pop("consider_pointer_prob_in_topk")
            else:
                consider_pointer_prob_in_topk = False

            if "pruning_token" in kwargs:
                pruning_token = kwargs.pop("pruning_token")
            else:
                pruning_token = 0

            if "pruning_pointer" in kwargs:
                pruning_pointer = kwargs.pop("pruning_pointer")
            else:
                pruning_pointer = 0

            # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
            self._validate_model_class()

            # priority: `generation_config` argument > `model.generation_config` (the default generation config)
            if generation_config is None:
                # legacy: users may modify the model configuration to control generation -- update the generation config
                # model attribute accordingly, if it was created from the model config
                if self.generation_config._from_model_config:
                    new_generation_config = GenerationConfig.from_model_config(self.config)
                    if new_generation_config != self.generation_config:
                        warnings.warn(
                            "You have modified the pretrained model configuration to control generation. This is a"
                            " deprecated strategy to control generation and will be removed soon, in a future version."
                            " Please use a generation configuration file (see"
                            " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                        )
                        self.generation_config = new_generation_config
                generation_config = self.generation_config

            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
            self._validate_model_kwargs(model_kwargs.copy())

            # 2. Set generation parameters if not already defined
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            pointer_logits_processor = (
                pointer_logits_processor if pointer_logits_processor is not None else LogitsProcessorList()
            )
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

            if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
                if model_kwargs.get("attention_mask", None) is None:
                    logger.warning(
                        "The attention mask and the pad token id were not set. As a consequence, you may observe "
                        "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                    )
                eos_token_id = generation_config.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_token_id = eos_token_id[0]
                logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
                generation_config.pad_token_id = eos_token_id

            # 3. Define model inputs
            # inputs_tensor has to be defined
            # model_input_name is defined if model-specific keyword input is passed
            # otherwise model_input_name is None
            # all model-specific keyword inputs are removed from `model_kwargs`
            inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
                inputs, generation_config.bos_token_id, model_kwargs
            )
            batch_size = inputs_tensor.shape[0]

            # 4. Define other model kwargs
            model_kwargs["output_attentions"] = generation_config.output_attentions
            model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
            model_kwargs["use_cache"] = generation_config.use_cache

            accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
            requires_attention_mask = "encoder_outputs" not in model_kwargs

            if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
                model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
                )

            # decoder-only models should use left-padding for generation
            if not self.config.is_encoder_decoder:
                if (
                    generation_config.pad_token_id is not None
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
                ):
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

            if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
                # if model is encoder decoder encoder_outputs are created
                # and added to `model_kwargs`
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name
                )

            # 5. Prepare `input_ids` which will be used for auto-regressive generation
            if self.config.is_encoder_decoder:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    batch_size,
                    decoder_start_token_id=generation_config.decoder_start_token_id,
                    bos_token_id=generation_config.bos_token_id,
                    model_kwargs=model_kwargs,
                    device=inputs_tensor.device,
                )
                pointer_ids = torch.full_like(input_ids, -100)
            else:
                raise NotImplementedError

            # 6. Prepare `max_length` depending on other stopping criteria.
            input_ids_seq_length = input_ids.shape[-1]
            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            if has_default_max_length and generation_config.max_new_tokens is None:
                warnings.warn(
                    "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
                    f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
                    " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
                    " recommend using `max_new_tokens` to control the maximum length of the generation.",
                    UserWarning,
                )
            elif has_default_max_length and generation_config.max_new_tokens is not None:
                generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            elif not has_default_max_length and generation_config.max_new_tokens is not None:
                raise ValueError(
                    "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                    " limit to the generated output length. Remove one of those arguments. Please refer to the"
                    " documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )

            if (
                generation_config.min_length is not None
                and generation_config.min_length > generation_config.max_length
            ):
                raise ValueError(
                    f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                    f" the maximum length ({generation_config.max_length})"
                )
            if input_ids_seq_length >= generation_config.max_length:
                input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
                logger.warning(
                    f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                    f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                    " increasing `max_new_tokens`."
                )

            # 7. determine generation mode
            is_constraint_gen_mode = (
                generation_config.constraints is not None or generation_config.force_words_ids is not None
            )

            is_contrastive_search_gen_mode = (
                generation_config.top_k is not None
                and generation_config.top_k > 1
                and generation_config.do_sample is False
                and generation_config.penalty_alpha is not None
                and generation_config.penalty_alpha > 0
            )

            is_greedy_gen_mode = (
                (generation_config.num_beams == 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
            )
            is_beam_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
            )
            is_beam_sample_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is True
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
            )
            # is_greedy_gen_mode = False
            # is_beam_gen_mode = True
            if generation_config.num_beam_groups > generation_config.num_beams:
                raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

            if self.device.type != input_ids.device.type:
                warnings.warn(
                    "You are calling .generate() with the `input_ids` being on a device type different"
                    f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                    f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                    " Please make sure that you have put `input_ids` to the"
                    f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                    " running `.generate()`.",
                    UserWarning,
                )

            # 8. prepare distribution pre_processing samplers
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
            )

            # 9. prepare stopping criteria
            stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=stopping_criteria
            )
            # 10. go into different generation modes
            if is_greedy_gen_mode:
                if generation_config.num_return_sequences > 1:
                    raise ValueError(
                        f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                        " greedy search."
                    )

                # 11. run greedy search
                return self.greedy_search(
                    input_ids,
                    pointer_ids,
                    logits_processor=logits_processor,
                    pointer_logits_processor=pointer_logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    **model_kwargs,
                )

            elif is_beam_gen_mode:
                if generation_config.num_return_sequences > generation_config.num_beams:
                    raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

                if stopping_criteria.max_length is None:
                    raise ValueError("`max_length` needs to be a stopping_criteria for now.")

                # 11. prepare beam search scorer
                beam_scorer = PAwareBeamSearchScorer(
                    batch_size=batch_size,
                    num_beams=generation_config.num_beams,
                    device=inputs_tensor.device,
                    length_penalty=generation_config.length_penalty,
                    do_early_stopping=generation_config.early_stopping,
                    num_beam_hyps_to_keep=generation_config.num_return_sequences,
                    pointer_scale=beam_search_pointer_scale,
                )
                # 12. interleave input_ids with `num_beams` additional sequences per batch
                input_ids, pointer_ids, model_kwargs = self._expand_inputs_for_generation(
                    input_ids=input_ids,
                    pointer_ids=pointer_ids,
                    expand_size=generation_config.num_beams,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    **model_kwargs,
                )
                # 13. run beam search
                return self.beam_search(
                    input_ids,
                    pointer_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    pointer_logits_processor=pointer_logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    consider_pointer_prob_in_topk=consider_pointer_prob_in_topk,
                    pruning_token=pruning_token,
                    pruning_pointer=pruning_pointer,
                    **model_kwargs,
                )

            elif is_beam_sample_gen_mode:
                # 11. prepare logits warper
                logits_warper = self._get_logits_warper(generation_config)

                if stopping_criteria.max_length is None:
                    raise ValueError("`max_length` needs to be a stopping_criteria for now.")
                # 12. prepare beam search scorer
                beam_scorer = PAwareBeamSearchScorer(
                    batch_size=batch_size * generation_config.num_return_sequences,
                    num_beams=generation_config.num_beams,
                    device=inputs_tensor.device,
                    length_penalty=generation_config.length_penalty,
                    do_early_stopping=generation_config.early_stopping,
                    pointer_scale=beam_search_pointer_scale,
                )

                # 13. interleave input_ids with `num_beams` additional sequences per batch
                input_ids, pointer_ids, model_kwargs = self._expand_inputs_for_generation(
                    input_ids=input_ids,
                    pointer_ids=pointer_ids,
                    expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    **model_kwargs,
                )

                # 14. run beam sample
                return self.beam_sample(
                    input_ids,
                    pointer_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    pointer_logits_processor=pointer_logits_processor,
                    logits_warper=logits_warper,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    consider_pointer_prob_in_topk=consider_pointer_prob_in_topk,
                    pruning_token=pruning_token,
                    pruning_pointer=pruning_pointer,
                    **model_kwargs,
                )
            raise NotImplementedError

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

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_pointer_logits = outputs.pointers[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            # next_pointer_logits = pointer_logits_processor(pointer_ids, next_pointer_logits)

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

            is_quote = (input_ids.unsqueeze(-1) == self.quoting.view([1] * input_ids.ndim + [-1])).any(-1)
            is_normal = self.normal_tokens.gather(0, input_ids.flatten()).view_as(input_ids)
            is_pointable = (is_quote.cumsum(1) % 2 == 0) & is_normal & (input_ids == next_tokens.unsqueeze(-1))
            is_pointable[:, 0] = 1
            is_in_quoting = is_quote.sum(1) % 2 == 1
            next_pointer_logits[~is_pointable] = float("-inf")  # torch.finfo(next_pointer_logits.dtype).min

            next_pointers = torch.argmax(next_pointer_logits, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            normal_token_mask = self.normal_tokens.gather(0, next_tokens)
            next_pointers[~normal_token_mask] = -100
            next_pointers[is_in_quoting] = -100

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
        consider_pointer_prob_in_topk=False,
        pruning_token=0,  # 0 means no pruning
        pruning_pointer=0,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
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

            is_quote = (input_ids.unsqueeze(-1) == self.quoting.view([1] * input_ids.ndim + [-1])).any(-1)
            # can only point to not quoted tokens
            tgt_is_pointable = is_quote.cumsum(1) % 2 == 0
            # can only point when not in quotes
            can_point_here = no_quote_here = tgt_is_pointable[:, -1]

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

            if consider_pointer_prob_in_topk:
                if (pt := pruning_token) > 0:
                    next_token_scores, selected_tokens = torch.topk(
                        next_token_scores, pt, dim=1, largest=True, sorted=False
                    )
                else:
                    pt = next_token_scores.shape[1]
                    selected_tokens = torch.arange(pt, device=next_token_scores.device)
                    selected_tokens = selected_tokens.expand_as(next_token_scores)

                # also need to be the same
                tgt_is_pointable = tgt_is_pointable[..., None] & (input_ids[..., None] == selected_tokens[:, None])
                # mask = self.normal_tokens.gather(0, input_ids.flatten()).view(input_ids.shape)
                # tgt_is_pointable = tgt_is_pointable & mask
                # tgt_is_pointable = tgt_is_pointable.unsqueeze(-1).expand(-1, -1, pt)
                # 0 is always pointable
                tgt_is_pointable[:, 0] = 1
                # also need to be normal tokens
                is_normal = self.normal_tokens.gather(0, selected_tokens.flatten()).view(selected_tokens.shape)
                can_point_here = can_point_here.unsqueeze(-1) & is_normal

                next_pointer_logits = next_pointer_logits.unsqueeze(-1).repeat(1, 1, pt)
                next_pointer_logits[~tgt_is_pointable] = float("-inf")
                next_pointer_scores = next_pointer_logits.log_softmax(1)

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

                next_pointer_scores.masked_fill_(~can_point_here.unsqueeze(1), float("-inf"))
                next_pointer_scores[:, 0].masked_fill_(~can_point_here, 0.0)

                next_token_scores = next_token_scores.unsqueeze(1) + next_pointer_scores
                # next_token_scores[next_token_scores == torch.finfo(next_token_scores.dtype).min] = float("-inf")
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

                # setup labels
                no_quote_here = no_quote_here.view(batch_size, -1)
                no_quote_here = no_quote_here.gather(1, next_indices)
                is_normal = self.normal_tokens.gather(0, next_tokens.flatten()).view(next_tokens.shape)
                mask = ~no_quote_here | ~is_normal
                next_pointers[mask] = -100

            else:
                raise NotImplementedError
                # reshape for beam search
                # vocab_size = next_token_scores.shape[-1]
                # next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                # # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
                # next_token_scores, next_tokens = torch.topk(
                #     next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                # )

                # next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                # next_tokens = next_tokens % vocab_size

                # mask = self.normal_tokens.gather(0, input_ids.flatten()).view(input_ids.shape)
                # tgt_is_pointable = tgt_is_pointable & mask
                # next_pointer_logits[~tgt_is_pointable] = float("-inf")

                # next_pointer_scores = next_pointer_logits.log_softmax(-1)
                # next_pointer_scores = next_pointer_scores.view(batch_size, num_beams, -1)

                # next_pointers_prob, next_pointers = next_pointer_logits.max(dim=-1)

                # next_pointers_prob = next_pointers_prob.repeat_interleave(2, -1)
                # next_pointers = next_pointers.repeat_interleave(2, -1)
                # is_quoted = is_quoted.gather(1, next_indices)

                # pointer_mask = is_quoted | ~self.normal_tokens.gather(0, next_tokens.flatten()).view(next_tokens.shape)
                # next_pointers[pointer_mask] = -100
                # next_pointers_prob[pointer_mask] = 0.0

                # next_token_scores += next_pointers_prob

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

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        pointer_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        pointer_logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        consider_pointer_prob_in_topk=False,
        pruning_token=0,  # 0 means no pruning
        pruning_pointer=0,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
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

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
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

            next_token_scores = logits_warper(input_ids, next_token_scores)

            is_quote = (input_ids.unsqueeze(-1) == self.quoting.view([1] * input_ids.ndim + [-1])).any(-1)
            # can only point to not quoted tokens
            tgt_is_pointable = is_quote.cumsum(1) % 2 == 0
            # can only point when not in quotes
            can_point_here = no_quote_here = tgt_is_pointable[:, -1]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logits_warper(input_ids, next_token_scores_processed),)
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

            # also need to be the same
            tgt_is_pointable = tgt_is_pointable[..., None] & (input_ids[..., None] == selected_tokens[:, None])
            # mask = self.normal_tokens.gather(0, input_ids.flatten()).view(input_ids.shape)
            # tgt_is_pointable = tgt_is_pointable & mask
            # tgt_is_pointable = tgt_is_pointable.unsqueeze(-1).expand(-1, -1, pt)
            # 0 is always pointable
            tgt_is_pointable[:, 0] = 1
            # also need to be normal tokens
            is_normal = self.normal_tokens.gather(0, selected_tokens.flatten()).view(selected_tokens.shape)
            can_point_here = can_point_here.unsqueeze(-1) & is_normal

            next_pointer_logits = next_pointer_logits.unsqueeze(-1).repeat(1, 1, pt)
            next_pointer_logits[~tgt_is_pointable] = float("-inf")
            next_pointer_scores = next_pointer_logits.log_softmax(1)

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

            next_pointer_scores.masked_fill_(~can_point_here.unsqueeze(1), float("-inf"))
            next_pointer_scores[:, 0].masked_fill_(~can_point_here, 0.0)

            next_token_scores = next_token_scores.unsqueeze(1) + next_pointer_scores
            # next_token_scores[next_token_scores == torch.finfo(next_token_scores.dtype).min] = float("-inf")
            next_token_scores = next_token_scores.view(batch_size, num_beams * pp * pt).clamp(
                torch.finfo(next_token_scores.dtype).min
            )
            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

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

            # setup labels
            no_quote_here = no_quote_here.view(batch_size, -1)
            no_quote_here = no_quote_here.gather(1, next_indices)
            is_normal = self.normal_tokens.gather(0, next_tokens.flatten()).view(next_tokens.shape)
            mask = ~no_quote_here | ~is_normal
            next_pointers[mask] = -100

            # vocab_size = next_token_scores.shape[-1]
            # next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # probs = nn.functional.softmax(next_token_scores, dim=-1)

            # next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            # next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            # next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            # next_tokens = torch.gather(next_tokens, -1, _indices)

            # next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            # next_tokens = next_tokens % vocab_size

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

            return PAwareBeamSampleOutput(
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

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )

        self.pointer_net._reorder_cache(beam_idx)
        return reordered_past

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        # do not re-init missing parameters. we have inited them in __init__
        # this may effect unexpected codes, but I did not find any usage except for the initialization
        return []
