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

import math
import random
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartAttention as OrigBartAttention
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration as OrigBartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.bart.modeling_bart import BartModel as OrigBartModel
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    _make_causal_mask,
)
from transformers.utils import logging

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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    if mask.ndim == 2:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    elif mask.ndim == 3:
        # add this to support TG's masking
        inverted_mask = 1.0 - mask.unsqueeze(1)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    elif mask.ndim == 4:
        # add this to support TG's masking
        inverted_mask = 1.0 - mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class BartAttention(OrigBartAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel."""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.tg_self_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # leave bias not shared
        if config.tg_args["tg_tie_q"]:
            self.tg_self_attn.q_proj.weight = self.self_attn.q_proj.weight
        if config.tg_args["tg_tie_k"]:
            self.tg_self_attn.k_proj.weight = self.self_attn.k_proj.weight
        if config.tg_args["tg_tie_v"]:
            self.tg_self_attn.v_proj.weight = self.self_attn.v_proj.weight

        self.use_external_feature = config.tg_args["use_external_feature"]
        if self.use_external_feature:
            self.external_feature_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        else:
            self.external_feature_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        if self_attn_past_key_value is None:
            past_key_value1, past_key_value2 = None, None
        else:
            past_key_value1, past_key_value2 = zip(*self_attn_past_key_value)
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states1, self_attn_weights1, present_key_value1 = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value1,
            attention_mask=attention_mask[0],
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states1 = nn.functional.dropout(hidden_states1, p=self.dropout, training=self.training)
        hidden_states2, self_attn_weights2, present_key_value2 = self.tg_self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value2,
            attention_mask=attention_mask[1],
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states2 = nn.functional.dropout(hidden_states2, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states1 + hidden_states2
        present_key_value = tuple(zip(present_key_value1, present_key_value2))

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            if self.use_external_feature:
                if cross_attn_past_key_value is None:
                    past_key_value1, past_key_value2 = None, None
                else:
                    past_key_value1, past_key_value2 = zip(*cross_attn_past_key_value)

                # encoder
                (hidden_states1, cross_attn_weights1, cross_attn_present_key_value1,) = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states[0],
                    attention_mask=encoder_attention_mask[0],
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value1,
                    output_attentions=output_attentions,
                )
                hidden_states1 = nn.functional.dropout(hidden_states1, p=self.dropout, training=self.training)

                # external feature, e.g. r2d2
                (hidden_states2, cross_attn_weights2, cross_attn_present_key_value2,) = self.external_feature_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states[1],
                    attention_mask=encoder_attention_mask[1],
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value2,
                    output_attentions=output_attentions,
                )
                hidden_states2 = nn.functional.dropout(hidden_states2, p=self.dropout, training=self.training)
                hidden_states = hidden_states1 + hidden_states2
                cross_attn_weights = (cross_attn_weights1, cross_attn_weights2)
                cross_attn_present_key_value = tuple(zip(cross_attn_present_key_value1, cross_attn_present_key_value2))
            else:
                (hidden_states, cross_attn_weights, cross_attn_present_key_value,) = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )

                hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartDecoder(BartPretrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        # if (tg_layers := config.tg_args["tg_layers"]) != "all":
        #     tg_disabled_config = deepcopy(config)
        #     tg_disabled_config.tg_args["tg_bypass_impl"] = 0
        #     self.layers = nn.ModuleList(
        #         [
        #             BartDecoderLayer(config if li in tg_layers else tg_disabled_config)
        #             for li in range(config.decoder_layers)
        #         ]
        #     )
        # else:
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)
        causal_attention_mask = combined_attention_mask

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            expanded_attn_mask = expanded_attn_mask.to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # import matplotlib.pyplot as plt
        # mat = combined_attention_mask[0][0].cpu().numpy()
        # plt.matshow(mat)
        # plt.colorbar()
        # plt.savefig('a.png')

        return causal_attention_mask, combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        causal_attention_mask, tg_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        attention_mask = (causal_attention_mask, tg_attention_mask)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if isinstance(encoder_attention_mask, tuple):
                encoder_attention_mask = (
                    _expand_mask(encoder_attention_mask[0], inputs_embeds.dtype, tgt_len=input_shape[-1]),
                    _expand_mask(encoder_attention_mask[1], inputs_embeds.dtype, tgt_len=input_shape[-1]),
                )  # TODO make this explicit
            else:
                encoder_attention_mask = _expand_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartModel(OrigBartModel):
    def __init__(self, config: BartConfig):
        BartPretrainedModel.__init__(self, config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


class BartForConditionalGeneration(OrigBartForConditionalGeneration):
    default_tg_args = {
        "use_external_feature": False,
        "tg_layers": "all",
        "tg_tie_q": False,
        "tg_tie_k": False,
        "tg_tie_v": False,
        "tg_fix_init": False,
    }

    def __init__(self, config: BartConfig, maskagent, tg_args):
        config.tg_args = self.default_tg_args | tg_args
        self.maskagent = maskagent

        BartPretrainedModel.__init__(self, config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.tg_args["tg_fix_init"]:
            logger.info("Fixing initialization")
            self.apply(self._init_weights)

    def generate(self, **kwargs):
        if self.maskagent.is_double_closing:
            self.tg_logits_processor = TransformerGrammarDoubleLogitsProcessor(self.maskagent.ranges)
        else:
            self.tg_logits_processor = TransformerGrammarSingleLogitsProcessor(self.maskagent.ranges)
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = []
        kwargs["logits_processor"].append(self.tg_logits_processor)
        return super().generate(**kwargs)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
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
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()

        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    (
                        past_state[0].index_select(0, beam_idx),
                        past_state[1].index_select(0, beam_idx),
                    )
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )

        beam_idx = beam_idx.tolist()
        self.py_mask_rules.reorder(beam_idx)
        self.tg_logits_processor.reorder(beam_idx)

        return reordered_past
