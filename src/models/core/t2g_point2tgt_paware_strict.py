import string
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
import smatch
import torch
import torch.nn as nn
from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer
from omegaconf import ListConfig, OmegaConf
from transformers import BartTokenizerFast
from transformers.models.bart.modeling_bart import ACT2FN, BartAttention
from transformers.models.bart.modeling_bart import (
    BartDecoderLayer as OrigBartDecoderLayer,
)

from src.models.components.extended_transformers import (
    bart_paware_strict,
    bart_paware_strict_adapter,
)
from src.models.components.extended_transformers.constrained_decoding import (
    ForcedNewConceptAfterOpeningLogit,
    ForcedSecondTokenLogitsProcessor,
)
from src.models.components.masking.utils import MaskAgent
from src.models.components.postprocess import (
    postprocess_attvar_double,
    postprocess_attvar_single,
)
from src.models.components.scalar_mix import ScalarMix
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CrossEntropy(nn.Module):
    def __init__(self, weight_0=None, label_smoothing=0.0) -> None:
        super().__init__()
        self.weight_0 = weight_0
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if self.weight_0 is not None:
            weight = torch.ones(input.shape[1], device=input.device)
            weight[0] = self.weight_0
        else:
            weight = None
        return torch.nn.functional.cross_entropy(input, target, weight=weight, label_smoothing=self.label_smoothing)


class Text2GraphPoint2TgtPAware(nn.Module):
    # Like StructBART. Reentrancies are represented as correference on target sequences.

    supported_models = {
        "bart_paware": bart_paware_strict.BartForConditionalGenerationAndPointer,
        "bart_paware_adapter": bart_paware_strict_adapter.BartForConditionalGenerationAndPointer,
    }

    def __init__(
        self,
        tokenizer,
        maskagent,
        model_name,
        path_to_pretrained,
        aligner,
        loss_strength=1.0,
        model_config=None,
        loss_weight_0=None,
        loss_label_smoothing=0.0,
        tg_args=None,
        pointer_args=None,
    ):
        super().__init__()
        self.tokenizer: BartTokenizerFast = tokenizer
        self.maskagent: MaskAgent = maskagent
        self.is_tg = maskagent.is_tg

        MODEL = self.supported_models[model_name]
        model_config = {} if model_config is None else model_config
        tg_args = {} if tg_args is None else OmegaConf.to_container(tg_args)
        pointer_args = {} if pointer_args is None else OmegaConf.to_container(pointer_args)
        if self.is_tg:
            self.model = MODEL.from_pretrained(
                path_to_pretrained,
                maskagent,
                tg_args,
                pointer_args,
                **model_config,
            )
        else:
            self.model = MODEL.from_pretrained(
                path_to_pretrained,
                pointer_args,
                **model_config,
            )

        self.aligner = None
        self.aligner_criteria = CrossEntropy(loss_weight_0, loss_label_smoothing)
        self.register_buffer("loss_strength", torch.tensor(loss_strength, dtype=torch.float))
        self.model_args = {}
        self.setup_aligner(aligner)

        self.model.pointer_net = self.aligner
        self.model.pointer_loss_fn = self.aligner_criteria
        self.model.tokenizer = self.tokenizer

        normal_tokens = [False] * len(self.tokenizer)
        for subtok, id in self.tokenizer.vocab.items():
            if (
                subtok.startswith("Ġ")
                and len(subtok) > 1
                and subtok[1] in string.ascii_letters
                and subtok not in ("Ġimperative", "Ġexpressive")
            ):
                normal_tokens[id] = True
        self.model.register_buffer("normal_tokens", torch.tensor(normal_tokens))
        quoting_tokens = [id for subtok, id in self.tokenizer.vocab.items() if '"' in subtok]
        if "<lit>" in self.tokenizer.vocab:
            quoting_tokens.append(self.tokenizer.convert_tokens_to_ids("<lit>"))
            quoting_tokens.append(self.tokenizer.convert_tokens_to_ids("</lit>"))
        self.model.register_buffer("quoting", torch.tensor(quoting_tokens))

    def forward(self, inputs):
        pointer_ids = torch.full_like(inputs["tgt_pt_pointer"], -100)
        pointer_ids[:, 1:] = inputs["tgt_pt_pointer"][:, :-1]

        # only if there are duplicate tokens, we need pointer to distinguish them
        mask = inputs["decoder_input_ids"][:, :, None] == inputs["decoder_input_ids"][:, None, :]
        mask = torch.tril(mask, -1)
        mask = mask.any(2)
        mask = torch.roll(mask, shifts=-1, dims=1)

        # ps = inputs["tgt_pt_pointer"][13].tolist()
        # seq = self.tokenizer.convert_ids_to_tokens(inputs["decoder_input_ids"][13])
        # for i, p in enumerate(ps[:-1]):
        #     if p >= 0:
        #         print(i + 1, '\t', seq[i + 1], '\t', seq[p])
        #     else:
        #         print(i+1, '\t', seq[i+1])

        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            labels=inputs["labels"],
            decoder_attention_mask=inputs["decoder_attention_mask"] if self.is_tg else None,
            pointer_ids=pointer_ids,
            pointer_labels=inputs["tgt_pt_pointer"],
            pointer_mask=~mask,
            **self.model_args,
        )
        return output

    def generate(self, inputs, **kwargs):
        kwargs.setdefault("max_new_tokens", 768)
        kwargs.setdefault("early_stopping", True)
        kwargs.setdefault("num_beams", 1)
        kwargs["no_repeat_ngram_size"] = 0
        kwargs["forced_bos_token_id"] = self.tokenizer.bos_token_id
        kwargs.update(self.model_args)

        output = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            logits_processor=[
                ForcedSecondTokenLogitsProcessor(self.maskagent.ranges.opening_non_terminal),
                ForcedNewConceptAfterOpeningLogit(
                    self.maskagent.ranges.opening_non_terminal, self.model.normal_tokens
                ),
            ],
            **kwargs,
        )

        aligned = output.pointers
        pointers_batch = aligned.tolist()
        ids_batch = []
        for bi, pointers in enumerate(pointers_batch):
            ids = list(range(len(pointers) - 1))
            for pi, pointer in enumerate(pointers[1:]):
                if pointer > 0 and pointer != pi:
                    ids[pi] = ids[pointer - 1]
                elif pointer < 0:
                    ids[pi] = None  # debug
            ids_batch.append(ids)

        # add dummy node to make token and vars matching
        tokens = [self.tokenizer.convert_ids_to_tokens(item) for item in output["sequences"]]
        pp_func = postprocess_attvar_double if self.maskagent.is_double_closing else postprocess_attvar_single
        output = list(map(pp_func, zip(tokens, ids_batch)))
        deserialized = list(map(lambda x: PenmanDeSerializer(x).get_graph_string(), output))
        snts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        output = [
            {"raw": o, "snt": s, "graph": g if (g is not None) and (g != "()") else "( b / bad )"}
            for o, s, g in zip(output, snts, deserialized, strict=True)
        ]
        return output

    def setup_aligner(self, args):
        aligner_type = args.pop("type")
        match aligner_type:
            case "bilinear":
                self.aligner = BilinearAligner(**args, input_size=self.model.config.d_model)
                self.model_args["output_hidden_states"] = True
            case "inplace_attn":
                self.aligner = InplaceAttentionAligner(**args, do_softmax=False)
                self.model_args["output_hidden_states"] = True
                self.model_args["output_attentions"] = True
            case "bypass_attn":
                self.aligner = BypassAttentionAligner(**args, do_softmax=False, model=self.model)
                self.model_args["output_hidden_states"] = True
                self.model_args["output_attentions"] = True
            case _:
                raise ValueError(f"Bad aligner type '{aligner_type}'")


class BilinearAligner(nn.Module):
    def __init__(self, input_size, hidden_size, bart_layers, var_format, deeper) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if deeper:
            mid_size = (self.input_size + self.hidden_size) // 2
            self.src_proj = nn.Sequential(
                nn.Linear(self.input_size, mid_size),
                nn.LeakyReLU(),
                nn.Linear(mid_size, self.hidden_size),
                nn.LeakyReLU(),
            )
            self.tgt_proj = nn.Sequential(
                nn.Linear(self.input_size, mid_size),
                nn.LeakyReLU(),
                nn.Linear(mid_size, self.hidden_size),
                nn.LeakyReLU(),
            )
        else:
            self.src_proj = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.LeakyReLU())
            self.tgt_proj = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.LeakyReLU())

        self.bilinear_weight = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        # target-side-pointer2: null to self, so no need to set offset
        assert var_format == "target-side-pointer"
        self.offset = 0 if var_format == "target-side-pointer2" else -1

        self.bart_layers = bart_layers
        self.scalar_mix = ScalarMix(bart_layers)

        self.generation_hidden_states_cache_h1 = None
        # self.generation_hidden_states_cache_h2 = None  # debug
        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def forward(self, model_output, pointer_mask=None):
        # pointer_mask: 1 for ok
        dec_h = model_output["decoder_hidden_states"][-self.bart_layers :]
        dec_h = self.scalar_mix(dec_h)
        h1 = self.src_proj(dec_h)
        h2 = self.tgt_proj(dec_h)
        if self.is_generating:
            h1, h2 = self.update_cache(h1, h2, model_output["decoder_hidden_states"][0])
        aligned = torch.einsum("qax,xy,qby->qba", h1, self.bilinear_weight, h2)

        if not self.is_generating:
            aligned = torch.where(
                torch.ones_like(aligned, dtype=torch.bool).tril(self.offset),
                aligned,
                float("-inf"),
            )
            if pointer_mask is not None:
                aligned[pointer_mask.unsqueeze(2).expand_as(aligned)] = float("-inf")
                aligned[..., 0][pointer_mask] = 0.0
        return aligned

    @contextmanager
    def generation_context(self):
        self.is_generating = True
        yield
        self.generation_hidden_states_cache_h1 = None
        # self.generation_hidden_states_cache_h2 = None
        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def update_cache(self, h1, h2, dec_emb):
        if self.generation_hidden_states_cache_h1 is not None:
            h1 = torch.cat([self.generation_hidden_states_cache_h1, h1], dim=1)
            # h2 = torch.cat([self.generation_hidden_states_cache_h2, h2], dim=1)
            dec_emb = torch.cat([self.generation_hidden_states_cache_l0, dec_emb], dim=1)
        self.generation_hidden_states_cache_h1 = h1
        # self.generation_hidden_states_cache_h2 = h2
        self.generation_hidden_states_cache_l0 = dec_emb
        return h1, h2

    def _reorder_cache(self, beam_idx):
        if self.generation_hidden_states_cache_h1 is not None:
            self.generation_hidden_states_cache_h1 = self.generation_hidden_states_cache_h1.index_select(0, beam_idx)
            # self.generation_hidden_states_cache_h2.index_select(0, beam_idx)
            self.generation_hidden_states_cache_l0 = self.generation_hidden_states_cache_l0.index_select(0, beam_idx)


class InplaceAttentionAligner(nn.Module):
    def __init__(self, method, bart_layers, bart_heads, do_softmax):
        super().__init__()
        assert method in ("mean", "max")
        self.method = method
        self.bart_layers = bart_layers if isinstance(bart_layers, (list, tuple, ListConfig)) else [bart_layers]
        self.bart_heads = bart_heads if bart_heads != "all" else None
        self.do_softmax = do_softmax

        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def forward(self, model_output, pointer_mask=None):
        if not self.is_generating:
            atts = model_output["decoder_attentions"]
            atts = torch.stack([atts[layer] for layer in self.bart_layers], dim=-1)
            att = self._reduce(atts, -1)
        else:  # generating
            self.update_cache(model_output["decoder_hidden_states"][0])
            atts = model_output["decoder_attentions"]
            atts = torch.stack([atts[layer] for layer in self.bart_layers], dim=-1)
            att = self._reduce(atts, -1)

        aligned = self._reduce(att[:, : self.bart_heads], 1)

        if not self.do_softmax:
            # att is already probabilities
            aligned = (aligned + 1e-9).log()

        if not self.is_generating:
            aligned = torch.where(
                torch.ones_like(aligned, dtype=torch.bool).tril(-1),
                aligned,
                float("-inf"),
            )
            if pointer_mask is not None:
                aligned[pointer_mask.unsqueeze(2).expand_as(aligned)] = float("-inf")
                aligned[..., 0][pointer_mask] = 0.0
        else:
            model_output["decoder_attentions"] = None  # avoid OOM

        return aligned

    def _reduce(self, tensor, dim):
        if self.method == "mean":
            return tensor.mean(dim)
        else:
            return tensor.amax(dim)

    @contextmanager
    def generation_context(self):
        self.is_generating = True
        yield
        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def update_cache(self, dec_emb):
        if self.generation_hidden_states_cache_l0 is not None:
            dec_emb = torch.cat([self.generation_hidden_states_cache_l0, dec_emb], dim=1)
        self.generation_hidden_states_cache_l0 = dec_emb

    def _reorder_cache(self, beam_idx):
        if self.generation_hidden_states_cache_l0 is not None:
            self.generation_hidden_states_cache_l0 = self.generation_hidden_states_cache_l0.index_select(0, beam_idx)


class BypassAttentionAligner(nn.Module):
    def __init__(self, method, bart_layers, do_softmax, model):
        super().__init__()
        assert method in ("mean", "max")
        self.method = method
        self.bart_layers = bart_layers if isinstance(bart_layers, (list, tuple, ListConfig)) else [bart_layers]
        self.do_softmax = do_softmax

        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

        if isinstance(model, bart_paware_strict.BartForConditionalGenerationAndPointer):
            parent_module = model.get_decoder().layers
            for layer in self.bart_layers:
                self._replace_module(parent_module, str(layer), BartDecoderLayer(model.config), parent_module[layer])
        else:
            for layer in model.get_decoder().layers:
                layer.return_bypass_attn = True

    def forward(self, model_output, pointer_mask=None):
        if not self.is_generating:
            atts = model_output["decoder_attentions"]
            atts = torch.stack([atts[layer] for layer in self.bart_layers], dim=-1)
            att = self._reduce(atts, -1)
        else:  # generating
            self.update_cache(model_output["decoder_hidden_states"][0])
            atts = model_output["decoder_attentions"]
            atts = torch.stack([atts[layer] for layer in self.bart_layers], dim=-1)
            att = self._reduce(atts, -1)

        aligned = self._reduce(att, 1)

        if not self.do_softmax:
            # att is already probabilities
            aligned = (aligned + 1e-9).log()

        if not self.is_generating:
            aligned = torch.where(
                torch.ones_like(aligned, dtype=torch.bool).tril(-1),
                aligned,
                float("-inf"),
            )
            if pointer_mask is not None:
                aligned[pointer_mask.unsqueeze(2).expand_as(aligned)] = float("-inf")
                aligned[..., 0][pointer_mask] = 0.0

        else:
            model_output["decoder_attentions"] = None  # avoid OOM

        return aligned

    def _reduce(self, tensor, dim):
        if self.method == "mean":
            return tensor.mean(dim)
        else:
            return tensor.amax(dim)

    @contextmanager
    def generation_context(self):
        self.is_generating = True
        yield
        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def update_cache(self, dec_emb):
        if self.generation_hidden_states_cache_l0 is not None:
            dec_emb = torch.cat([self.generation_hidden_states_cache_l0, dec_emb], dim=1)
        self.generation_hidden_states_cache_l0 = dec_emb

    def _reorder_cache(self, beam_idx):
        if self.generation_hidden_states_cache_l0 is not None:
            self.generation_hidden_states_cache_l0 = self.generation_hidden_states_cache_l0.index_select(0, beam_idx)

    @staticmethod
    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)


class BartDecoderLayer(OrigBartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)

        self.pointer_self_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.pointer_self_attn.k_proj.weight = self.self_attn.k_proj.weight
        self.pointer_self_attn.k_proj.bias = self.self_attn.k_proj.bias

        self.pointer_self_attn.v_proj.weight = self.self_attn.v_proj.weight
        self.pointer_self_attn.v_proj.bias = self.self_attn.v_proj.bias

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
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states1, self_attn_weights1, present_key_value1 = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states1 = nn.functional.dropout(hidden_states1, p=self.dropout, training=self.training)
        hidden_states2, self_attn_weights2, present_key_value2 = self.pointer_self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states2 = nn.functional.dropout(hidden_states2, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states1 + hidden_states2
        present_key_value = present_key_value1

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

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
            outputs += (self_attn_weights2, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
