import string

import torch
import torch.nn as nn
from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer
from transformers import BartTokenizerFast
from transformers.models.bart import modeling_bart

from src.models.components.extended_transformers.constrained_decoding import (
    ForcedNewConceptAfterOpeningLogit,
)
from src.models.components.postprocess import postprocess_attvar_single
from src.models.components.scalar_mix import ScalarMix
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Text2GraphPoint2Tgt(nn.Module):
    # Like StructBART. Reentrancies are represented as correference on target sequences
    # Only support bart because most previous tokens are masked due to TG maskrules.
    supported_models = {
        "bart": modeling_bart.BartForConditionalGeneration,
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
        **tg_args,
    ):
        super().__init__()
        assert not maskagent.is_tg
        self.tokenizer: BartTokenizerFast = tokenizer
        self.maskagent = maskagent
        model_config = {} if model_config is None else model_config
        self.model = self.supported_models[model_name].from_pretrained(path_to_pretrained, **model_config)

        self.aligner = None
        self.aligner_criteria = nn.CrossEntropyLoss()
        self.register_buffer("loss_strength", torch.tensor(loss_strength, dtype=torch.float))
        self.model_args = {}
        self.setup_aligner(aligner)

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
        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            labels=inputs["labels"],
            **self.model_args,
        )
        aligned = self.aligner(output, is_forward=True)
        additional_loss = self.aligner_criteria(aligned.transpose(1, 2), inputs["tgt_pt_pointer"])
        output.logs = {"ml_loss": output.loss, "pointer_loss": additional_loss}
        output.loss = output.loss + additional_loss * self.loss_strength
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

        aligned = self.aligner(output, is_forward=False)
        tokens = [self.tokenizer.convert_ids_to_tokens(item) for item in output["sequences"]]
        output = list(map(postprocess_attvar_single, zip(tokens, aligned)))
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
        self.offset = 0 if var_format == "target-side-pointer2" else -1

        self.bart_layers = bart_layers
        self.scalar_mix = ScalarMix(bart_layers)

    def forward(self, model_output, is_forward):

        if is_forward:
            dec_h = model_output["decoder_hidden_states"][-self.bart_layers :]
            dec_h = self.scalar_mix(dec_h)
            h1 = self.src_proj(dec_h)
            h2 = self.tgt_proj(dec_h)
            aligned = torch.einsum("qax,xy,qby->qba", h1, self.bilinear_weight, h2)
        else:
            dec_h = [item[-self.bart_layers :] for item in model_output["decoder_hidden_states"]]
            dec_h = [torch.cat([item[i] for item in dec_h], dim=1) for i in range(len(dec_h[0]))]
            dec_h = self.scalar_mix(dec_h)
            dec_h = dec_h.view(len(model_output["sequences"]), -1, *dec_h.shape[1:])[:, 0]
            h1 = self.src_proj(dec_h)
            h2 = self.tgt_proj(dec_h)
            aligned = torch.einsum("qax,xy,qby->qba", h1, self.bilinear_weight, h2)

        aligned = torch.where(torch.ones_like(aligned, dtype=torch.bool).tril(self.offset), aligned, float("-inf"))
        if is_forward:
            return aligned
        else:

            # we first assign unique id to all tokens, then set equality according to the alignment
            pointers_batch = aligned.argmax(2).tolist()
            ids_batch = []
            for bi, pointers in enumerate(pointers_batch):
                ids = list(range(len(pointers)))
                for pi, pointer in enumerate(pointers):
                    if pointer > 0 and pointer != pi:
                        ids[pi] = ids[pointer]
                ids_batch.append(ids)
            return ids_batch
