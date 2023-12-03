import argparse
import json
import os
import re
from collections import defaultdict

from amrlib.models.parse_xfm.penman_serializer import (
    PenmanDeSerializer,
    TType,
    load_and_serialize,
)
from joblib import Memory
from tqdm import tqdm
from transformers import AutoTokenizer

location = "./logs/cachedir"
memory = Memory(location, verbose=0)

load_and_serialize = memory.cache(load_and_serialize)

reentry_pattern = re.compile(r"^(.*)_(\d+)$")

parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--model", default="facebook/bart-large")
parser.add_argument("--data-dir", default="data/AMR3.0/tdata_xfm")
parser.add_argument("--train-file", default="train.txt.nowiki")
parser.add_argument("--dev-file", default="dev.txt.nowiki")
parser.add_argument("--test-file", default="test.txt.nowiki")
parser.add_argument("--normalize-url", action="store_true")
parser.add_argument("--remove-len-1-span", action="store_true")
parser.add_argument("--detach-var", action="store_true")
parser.add_argument("--add-subword-grouping", action="store_true")
parser.add_argument("--replace-quote", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)


def is_num(val):
    if val in ("nan", "infinity"):
        return False
    try:
        x = float(val)
        return True
    except ValueError:
        return False


def clean(e):
    lines = [line.strip() for line in e.splitlines()]
    lines = [line for line in lines if (line and not line.startswith("#"))]
    string = " ".join(lines)
    string = string.replace("\t", " ")  # replace tabs with a space
    string = re.sub(" +", " ", string)  # squeeze multiple spaces into a single
    return string


def normalize_url(serial, sent, normalized_form="http://link"):
    # NOTE: this may change the AMR because in some rare cases, there are multiple constants
    #   corresponding to the same url.
    url = None
    while result := re.search(r'"https?:\/\/[^ ]*"', serial):
        url = result.group()
        serial = serial.replace(url, "@URL@")
        sent = sent.replace(url[1:-1], "@URL@")
    if url is not None:
        serial = serial.replace("@URL@", f'"{normalized_form}"')
        sent = sent.replace("@URL@", normalized_form)
    return serial, sent, url


#########################
# code for sanity check


def token_type(token):
    if token in {"(", ")"}:
        return TType.paren
    elif token.startswith(":"):
        return TType.role
    elif token in {"-", "+"}:  # 'interrogative', 'imperative', 'expressive']): could be nodes
        return TType.attrib  # instead of including here, test in logic above
    elif token.startswith('"') or token.endswith('"') or token[0].isdigit():  # fault tolerant def
        return TType.attrib
    elif is_num(token):
        return TType.attrib
    elif token == "/":
        return TType.sep
    else:
        return TType.concept


def add_bracket_to_single(text):
    # return text
    tokens = PenmanDeSerializer.graph_tokenize(text)
    processed = []
    allocated = set()  # bracket means allocating new variables
    for token in tokens:
        ttype = token_type(token)
        if ttype == TType.concept:
            if token not in ("interrogative", "imperative", "expressive") and token not in allocated:
                allocated.add(token)
                if len(processed) > 0 and processed[-1] != "(":
                    processed.extend(["(", token, ")"])
                else:
                    processed.append(token)
                continue
        processed.append(token)
    return " ".join(processed)


# end
#########################


def transform(path):
    data = load_and_serialize(path, False)

    converted = []
    for di, (serial, sent) in tqdm(enumerate(zip(data["serials"], data["sents"]))):

        if args.remove_len_1_span:
            # ref_serial = serial

            serial = re.sub(r"(?<!^)\( ([^ ]+) \)", r"\1", serial)
            # recover = add_bracket_to_single(serial)
            # if ref_serial != recover:
            #     breakpoint()

        url = None
        if args.normalize_url:
            serial, sent, url = normalize_url(serial, sent)

        processed_serial = []
        variables = []

        if args.replace_quote:
            num_quote = serial.count('"')
            serial_modified = re.sub(' "', "<lit>", serial)
            serial_modified = re.sub('"', "</lit>", serial_modified)
            tokens = serial_modified.split()
        else:
            tokens = serial.split()

        for i, c in enumerate(tokens):
            if c == "(":
                processed_serial.append("(99")  # match the special token defined in prepare_bart.py
                variables.append(None)
            elif c == ")":
                processed_serial.append("99)")
                variables.append(None)
            elif c[0] == ":" and len(c) > 1:  # rel
                processed_serial.append(c)
                variables.append(None)
            else:
                # variable or constant
                if args.detach_var:
                    if c[0] == '"' or c in ("-", "+", "interrogative", "imperative", "expressive") or is_num(c):
                        variables.append(None)
                        processed_serial.append(c)
                    else:
                        match = re.match(reentry_pattern, c)
                        if match is None:
                            variables.append(-1)
                            processed_serial.append(c)
                        else:
                            surface, id_ = match.groups()
                            variables.append(int(id_))
                            processed_serial.append(surface)
                else:
                    variables.append(None)
                    processed_serial.append(c)

        tgt = " " + " ".join(processed_serial)  # make sure the first ( is tokenized correctly

        # print(tokenizer.tokenize(tgt))
        # print(tokenizer.decode(tokenizer(tgt).input_ids[1:-1]))
        # print(tgt == tokenizer.decode(tokenizer(tgt).input_ids[1:-1]))
        # breakpoint()

        src_processed = tokenizer(sent, return_attention_mask=False, truncation=True)
        tgt_processed = tokenizer(
            text_target=tgt,
            truncation=True,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )

        if args.replace_quote:
            _tokens = tokenizer.convert_ids_to_tokens(tgt_processed["input_ids"])
            _left = _tokens.count("<lit>")
            _right = _tokens.count("</lit>")
            assert _left == _right and (_left + _right) == num_quote, (_left, _right, num_quote)
        # # check quote mark
        # quote_char_cnt = tgt.count('"')
        # if quote_char_cnt % 2 != 0:
        #     breakpoint()

        # left_quote = tgt_processed['input_ids'].count(22)
        # right_quote = tgt_processed['input_ids'].count(113)
        # if left_quote != right_quote or left_quote + right_quote != quote_char_cnt:
        #     breakpoint()

        variables_processed = {}
        if args.detach_var:

            # find positions of all beginning characters
            token_start_position = []
            for i, c in enumerate(tgt):
                if c == " ":
                    token_start_position.append(i + 1)

            # find word's beginning subword position
            cursor = 0
            word2start_subtok = []
            for i, (s, e) in enumerate(tgt_processed["offset_mapping"][1:-1], start=1):  # skip bos and eos
                # truncation
                if cursor >= len(token_start_position):
                    break
                # ideally, s == token_start_position[cursor], beginning char is the beginning of the subtok
                if s <= token_start_position[cursor] < e:
                    if s != token_start_position[cursor]:
                        print("[Bad matching 1]")
                    word2start_subtok.append(i)
                    cursor += 1
                # unlikely
                elif token_start_position[cursor] < s:
                    raise RuntimeError

            # build a dict { surface-form : { group : [ subtok_index ] } }
            subtoks = tokenizer.convert_ids_to_tokens(tgt_processed["input_ids"])
            for i, v in enumerate(variables):
                if v is not None and i < len(word2start_subtok):  # maybe truncated
                    form = processed_serial[i]
                    subtok_i = word2start_subtok[i]
                    if form not in variables_processed:
                        variables_processed[form] = {v: [subtok_i]}
                    else:
                        if v not in variables_processed[form]:
                            variables_processed[form][v] = [subtok_i]
                        else:
                            variables_processed[form][v].append(subtok_i)

                    # sanity check: subtok match surface form
                    assert subtoks[subtok_i].lstrip("Ġ ") in form, (
                        subtoks[subtok_i].lstrip("Ġ "),
                        form,
                    )
                    if not form.startswith(subtoks[subtok_i].lstrip("Ġ ")):
                        print("[Bad matching 2]", subtoks[subtok_i].lstrip("Ġ "), form)

            # sanity check: use dict to build the original serial sequence
            recovery = tokenizer.decode(tgt_processed["input_ids"][1:-1])
            recovery = recovery.split()
            start_subtok2word = {l: i for i, l in enumerate(word2start_subtok)}
            for groups in variables_processed.values():
                for group, locations in groups.items():
                    if group != -1:
                        for loc in locations:
                            loc = start_subtok2word[loc]
                            recovery[loc] = recovery[loc] + f"_{group}"
            recovery = " ".join(recovery)
            recovery = re.sub(r"\(99", "(", recovery)
            recovery = re.sub(r"99\)", ")", recovery)

            if args.replace_quote:
                recovery = re.sub("<lit>", ' "', recovery)
                recovery = re.sub("</lit>", '"', recovery)

            # just check startswith because of the possibility of truncation
            if not serial.startswith(recovery):
                print()
                print(recovery)
                print(serial)
                # breakpoint()
                raise RuntimeError

            if args.add_subword_grouping:
                # find all following tokens
                start_subtokens = set(word2start_subtok)
                start_subtoken_i = None
                start2all = defaultdict(list)
                for i, (s, e) in enumerate(tgt_processed["offset_mapping"][1:-1], start=1):
                    if i in start_subtokens:
                        start_subtoken_i = i
                    elif start_subtoken_i is None:
                        continue
                    else:
                        start2all[start_subtoken_i].append(i)
                # print(variables_processed)
                for groups in variables_processed.values():
                    for locations in groups.values():
                        for loc in locations.copy():
                            if len(start2all[loc]) > 0:
                                locations.extend(start2all[loc])
                # print(variables_processed)

        converted.append(
            (
                src_processed["input_ids"],
                tgt_processed["input_ids"],
                variables_processed,
                url,
                data["sents"][di],
                clean(data["graphs"][di]),
            )
        )

    return converted


def save(data, path):
    f = open(path + ".tsv", "w")
    f_snt = open(path + ".snt", "w")
    f_graph = open(path + ".graph", "w")
    if args.normalize_url:
        f_url = open(path + ".url", "w")
    else:
        f_url = None
    for sids, tids, vars, url, sent, graph in data:
        f.write(f'{",".join(map(str, sids))}\t{",".join(map(str, tids))}\t{json.dumps(vars,ensure_ascii=False)}\n')
        f_snt.write(f"{sent}\n")
        f_graph.write(f"{graph}\n")
        if args.normalize_url:
            f_url.write("-\n" if url is None else f"{url[1:-1]}\n")

    f.close()
    f_snt.close()
    f_graph.close()
    if f_url is not None:
        f_url.close()


os.makedirs(args.output_dir, exist_ok=True)
data = transform(os.path.join(args.data_dir, args.dev_file))
save(data, os.path.join(args.output_dir, "dev"))
data = transform(os.path.join(args.data_dir, args.test_file))
save(data, os.path.join(args.output_dir, "test"))
data = transform(os.path.join(args.data_dir, args.train_file))
save(data, os.path.join(args.output_dir, "train"))
