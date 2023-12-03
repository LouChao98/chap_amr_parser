import argparse
import re
from typing import List

import penman
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
)

parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--model", default="facebook/bart-large")
parser.add_argument("--train_file", default="data/AMR3.0/tdata_xfm/train.txt.nowiki")
parser.add_argument("--add-amr-tokens", action="store_true")
parser.add_argument("--add-lit-tokens", action="store_true")
parser.add_argument("--add-bibl-tokens", action="store_true")
args = parser.parse_args()

# collect graph

graphs: List[penman.Graph] = []
for i, graph in enumerate(penman.iterdecode(open(args.train_file))):
    snt = graph.metadata["snt"]
    graphs.append(graph)

# collect special tokens and senses
if args.add_amr_tokens:
    rels = set()
    senses = set()
    for graph in graphs:
        for triple in graph.triples:
            rels.add(triple[1])
            if (g := re.match(r"[\w-]*(-\d+)", triple[2])) is not None:
                senses.add(g[1])

    rels = [f" {item}" for item in rels]
    senses = list(senses) + ["-of"]
    rels.sort()
    senses.sort()
else:
    rels, senses = [], []

if args.add_lit_tokens:
    extra = ["<lit>", "</lit>"]
else:
    extra = []

if args.add_bibl_tokens:
    extra += ['[GEN]', '[ANS]']

# adjust embedding

bart: BartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(args.model)
special_start_idx = bart.config.vocab_size
tokenizer = AutoTokenizer.from_pretrained(
    args.model, additional_special_tokens=rels + senses + extra + [" (99", " 99)"]
)
bart.resize_token_embeddings(len(tokenizer))
emb = bart.model.shared.weight.data

rel_shared_emb = torch.randn(bart.config.d_model) * 0.001
sense_shared_emb = torch.randn(bart.config.d_model) * 0.001
opening_nt_shared_emb = torch.randn(bart.config.d_model) * 0.001
closing_nt_shared_emb = torch.randn(bart.config.d_model) * 0.001


with torch.no_grad():

    # setup embedding for relations
    for i, tok in enumerate(rels, start=special_start_idx):
        assert tokenizer.convert_tokens_to_ids(tok) == i
        pieces = tokenizer(tok[2:].lower(), add_special_tokens=False)
        emb[i] = sum(emb[j] for j in pieces["input_ids"]) / len(pieces["input_ids"]) + rel_shared_emb

    # setup embedding for sense marks
    for i, tok in enumerate(senses, start=special_start_idx + len(rels)):
        assert tokenizer.convert_tokens_to_ids(tok) == i
        pieces = tokenizer(tok[1:], add_special_tokens=False)
        emb[i] = sum(emb[j] for j in pieces["input_ids"]) / len(pieces["input_ids"]) + sense_shared_emb

    if args.add_lit_tokens:
        # setup embedding for left "
        emb[tokenizer.convert_tokens_to_ids('<lit>')] = emb[tokenizer.convert_tokens_to_ids('Ġ"')]

        # setup embedding for right "
        emb[tokenizer.convert_tokens_to_ids('</lit>')] = emb[tokenizer.convert_tokens_to_ids('"')]
    
    if args.add_bibl_tokens:
         emb[tokenizer.convert_tokens_to_ids('[GEN]')] = torch.randn(bart.config.d_model)
         emb[tokenizer.convert_tokens_to_ids('[ANS]')] = torch.randn(bart.config.d_model)

    # setup embedding for opening NT
    emb[-2] = emb[tokenizer.convert_tokens_to_ids("Ġ(")] + opening_nt_shared_emb

    # setup embedding for closing NT
    emb[-1] = emb[tokenizer.convert_tokens_to_ids("Ġ)")] + closing_nt_shared_emb


tokenizer.save_pretrained(args.output_dir)
bart.save_pretrained(args.output_dir)


# import torch
# from transformers import AutoModel, AutoTokenizer


# m1 = AutoModel.from_pretrained('facebook/bart-base')
# m2 = AutoModel.from_pretrained('data/AMR/bart-base_processed')
# t = AutoTokenizer.from_pretrained('data/AMR/bart-base_processed')

# text = 'The forest raven (Corvus tasmanicus), or Tasmanian raven, is a passerine bird in the family Corvidae native to Tasmania and parts of southern Victoria and New South Wales.'
# inp = t(text, return_tensors='pt')

# with torch.no_grad():
#     o1 = m1(**inp)
#     o2 = m2(**inp)

# assert torch.allclose(o1.last_hidden_state, o2.last_hidden_state)
# assert torch.allclose(o1.encoder_last_hidden_state, o2.encoder_last_hidden_state)
