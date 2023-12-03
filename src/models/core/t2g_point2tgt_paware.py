import math
import string
from contextlib import contextmanager

import torch
import torch.nn as nn
from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer
from omegaconf import OmegaConf
from transformers import BartTokenizerFast

from src.models.components.extended_transformers import bart_paware, bart_paware_adapter
from src.models.components.extended_transformers.constrained_decoding import (
    ForcedNewConceptAfterOpeningLogit,
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
        "bart_paware": bart_paware.BartForConditionalGenerationAndPointer,
        "bart_paware_adapter": bart_paware_adapter.BartForConditionalGenerationAndPointer,
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
        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            labels=inputs["labels"],
            decoder_attention_mask=inputs["decoder_attention_mask"] if self.is_tg else None,
            pointer_ids=pointer_ids,
            pointer_labels=inputs["tgt_pt_pointer"],
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
                ForcedNewConceptAfterOpeningLogit(self.maskagent.ranges.opening_non_terminal, self.model.normal_tokens)
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
            case _:
                raise ValueError(f"Bad aligner type '{aligner_type}'")


class BilinearAligner(nn.Module):
    def __init__(self, input_size, hidden_size, bart_layers, var_format, deeper):
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
        self.offset = 0 if var_format == "target-side-pointer2" else -1

        self.bart_layers = bart_layers
        self.scalar_mix = ScalarMix(bart_layers)

        self.generation_hidden_states_cache_h1 = None
        # self.generation_hidden_states_cache_h2 = None  # debug
        self.generation_hidden_states_cache_l0 = None
        self.is_generating = False

    def forward(self, model_output):

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
