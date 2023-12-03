import argparse
import json

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("model")
parser.add_argument("file")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
data = []
with open(args.file) as f:
    for i, line in enumerate(f):
        sids, tids, variables = line.split("\t", 2)
        sids = list(map(int, sids.split(",")))
        tids = list(map(int, tids.split(",")))
        # variables = json.loads(variables)
        inst = {"id": i, "src": sids, "tgt": tids, "var": variables}
        data.append(inst)


def show(inst, num=20):
    print("src:  ", tokenizer.decode(inst["src"]))
    print("stok: ", tokenizer.convert_ids_to_tokens(inst["src"])[:num])
    print("tgt: ", tokenizer.decode(inst["tgt"]))
    print("ttok: ", tokenizer.convert_ids_to_tokens(inst["tgt"])[:num])
