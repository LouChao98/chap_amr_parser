from contextlib import contextmanager

import torch
import torch.nn as nn
from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer
from omegaconf import ListConfig, OmegaConf
from transformers import BartTokenizerFast, RepetitionPenaltyLogitsProcessor

from src.models.components.extended_transformers import (
    bart_adapter_closing_only_pointer,
)
from src.models.components.masking.utils import MaskAgent
from src.models.components.postprocess import postprocess_inline_closing_only
from src.models.components.scalar_mix import ScalarMix
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphInline(nn.Module):
    supported_models = {
        "bart_adapter": bart_adapter_closing_only_pointer.BartForConditionalGeneration,
    }

    def __init__(
        self,
        tokenizer,
        maskagent,
        model_name,
        path_to_pretrained,
        aligner,
        model_config=None,
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
            tg_args = MODEL.default_tg_args | tg_args
            log.info(f"Setup {model_name} with args {tg_args=}, {model_config=}")
            self.model = MODEL.from_pretrained(path_to_pretrained, maskagent, tg_args, pointer_args, **model_config)
        else:
            raise NotImplementedError
            self.model = MODEL.from_pretrained(path_to_pretrained, pointer_args, **model_config)

        self.aligner = None
        self.aligner_criteria = torch.nn.CrossEntropyLoss()
        self.model_args = {}
        self.setup_aligner(aligner)

        self.model.pointer_net = self.aligner
        self.model.pointer_loss_fn = self.aligner_criteria
        self.model.tokenizer = self.tokenizer

        normal_tokens = [False] * len(self.tokenizer)
        for subtok, id in self.tokenizer.vocab.items():
            if subtok.startswith("Ġ") and len(subtok) > 1:
                normal_tokens[id] = True
        self.model.register_buffer("normal_tokens", torch.tensor(normal_tokens).cuda())

    def forward(self, inputs):
        # only tokens before normal tokens are allowed to be pointed.
        shape = inputs["decoder_input_ids"].shape
        is_normal_token = self.model.normal_tokens.gather(0, inputs["decoder_input_ids"].flatten()).view(shape)
        is_normal_token[:, :-1] = is_normal_token[:, 1:].clone()

        pointer_masks = inputs["closing_pointer_mask"]
        pointer_masks = pointer_masks & is_normal_token.unsqueeze(1)

        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"] if self.is_tg else None,
            labels=inputs["labels"],
            closing_pointers=inputs["closing_pointers"],
            closing_pointer_mask=inputs["closing_pointer_mask"] == 0,
            **self.model_args,
        )
        return output

    def generate(self, inputs, **kwargs):
        kwargs.setdefault("max_new_tokens", 370)
        kwargs.setdefault("early_stopping", True)
        kwargs.setdefault("num_beams", 1)
        kwargs["no_repeat_ngram_size"] = 0
        kwargs["forced_bos_token_id"] = self.tokenizer.bos_token_id
        kwargs.update(self.model_args)

        output = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # pointer_logits_processor=RepetitionPenaltyLogitsProcessor(0.1),
            return_dict_in_generate=True,
            **kwargs,
        )
        pointers = output.pointers.tolist()
        tokens = [self.tokenizer.convert_ids_to_tokens(item) for item in output["sequences"]]

        pp_func = postprocess_inline_closing_only
        output = list(map(pp_func, zip(tokens, pointers)))
        deserialized = list(map(lambda x: PenmanDeSerializer(x).get_graph_string(), output))
        snts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

        output = [
            {"raw": o, "snt": s, "graph": g if g is not None and g != "()" else "( b / bad )"}
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
            case _:
                raise ValueError(f"Bad aligner type '{aligner_type}'")


class BilinearAligner(nn.Module):
    def __init__(self, input_size, hidden_size, bart_layers, deeper) -> None:
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

        self.bart_layers = bart_layers
        self.scalar_mix = ScalarMix(bart_layers)

        self.generation_hidden_states_cache_h1 = None
        self.is_generating = False

    def forward(self, model_output, pointer_mask):
        # pointer_mask: 1 for ok
        dec_h = model_output["decoder_hidden_states"][-self.bart_layers :]
        dec_h = self.scalar_mix(dec_h)

        # print(dec_h.flatten(1).sum(1))

        h1 = self.src_proj(dec_h)
        h2 = self.tgt_proj(dec_h)
        if self.is_generating:
            h1, h2 = self.update_cache(h1, h2)
        aligned = torch.einsum("qax,xy,qby->qba", h1, self.bilinear_weight, h2)

        aligned[pointer_mask.expand_as(aligned)] = float("-inf")
        return aligned

    @contextmanager
    def generation_context(self):
        self.is_generating = True
        yield
        self.generation_hidden_states_cache_h1 = None
        self.is_generating = False

    def update_cache(self, h1, h2):
        if self.generation_hidden_states_cache_h1 is not None:
            h1 = torch.cat([self.generation_hidden_states_cache_h1, h1], dim=1)
            # h2 = torch.cat([self.generation_hidden_states_cache_h2, h2], dim=1)

        self.generation_hidden_states_cache_h1 = h1
        # self.generation_hidden_states_cache_h2 = h2
        return h1, h2

    def _reorder_cache(self, beam_idx):
        self.generation_hidden_states_cache_h1 = self.generation_hidden_states_cache_h1.index_select(0, beam_idx)


class InplaceAttentionAligner(nn.Module):
    def __init__(self, method, bart_layers, bart_heads, do_softmax, **kwargs):
        log.warning(f"Unexpected args {kwargs}")
        super().__init__()
        assert method in ("mean", "max")
        self.method = method
        self.bart_layers = bart_layers if isinstance(bart_layers, (list, tuple, ListConfig)) else [bart_layers]
        self.bart_heads = bart_heads if bart_heads != "all" else None
        self.do_softmax = do_softmax

        self.is_generating = False

    def forward(self, model_output, pointer_mask=None):
        if not self.is_generating:
            atts = model_output["decoder_attentions"]
            atts = torch.stack([atts[layer] for layer in self.bart_layers], dim=-1)
            att = self._reduce(atts, -1)
        else:  # generating
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
            aligned[pointer_mask.expand_as(aligned)] = float("-inf")
        else:
            aligned[pointer_mask.expand_as(aligned)] = float("-inf")
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
        self.is_generating = False

    def _reorder_cache(self, beam_idx):
        pass
