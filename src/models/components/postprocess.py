import re

from amrlib.models.parse_xfm.penman_serializer import PenmanDeSerializer, TType

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def postprocess_inline_double(text):
    text = re.sub(r"<(\/s|s|pad)>", "", text)
    text = re.sub(r"99\) 99\)", ")", text)
    text = re.sub(r"99\)", ")", text)  # remove or to )
    text = re.sub(r"\(99", "(", text)
    text = re.sub(r"</?graph>", "", text)
    text = re.sub("<lit>", ' "', text)
    text = re.sub("</lit>", '"', text)
    text = add_bracket_to_single(text)
    return text


def postprocess_inline_single(text):
    text = re.sub(r"<(\/s|s|pad)>", "", text)
    text = re.sub(r"99\)", ")", text)
    text = re.sub(r"\(99", "(", text)
    text = re.sub(r"</?graph>", "", text)
    text = re.sub("<lit>", ' "', text)
    text = re.sub("</lit>", '"', text)
    text = add_bracket_to_single(text)
    return text


def postprocess_inline_closing_only(data):
    text, pointers = data
    text = text
    num_to_add = [0] * len(text)
    for t, p in zip(text, pointers):
        if t == " 99)":
            num_to_add[p] += 1

    processed = []
    for t, n in zip(text, num_to_add):
        processed.append(t)
        if n > 0:
            processed.extend(" (" for _ in range(n))

    # breakpoint()
    text = "".join(processed).replace("Ġ", " ")
    text = re.sub(r"<(\/s|s|pad)>", "", text)
    text = re.sub(r"99\)", ")", text)
    text = re.sub(r"</?graph>", "", text)
    text = re.sub("<lit>", ' "', text)
    text = re.sub("</lit>", '"', text)
    text = add_bracket_to_single(text)
    return text


def postprocess_attvar_double(data):
    tokens, vars = data
    tokens = tokens[2:]  # skip special tokens
    vars = vars[1:]
    _t, _v = [], []
    skip = False
    for t, v in zip(tokens, vars):
        if t in ("<s>", "<pad>", "</s>"):
            continue
        if skip:
            skip = False
            continue
        if t == " 99)":
            skip = True
        elif t == "<lit>":
            t = ' "'
        elif t == "</lit>":
            t = '"'
        _t.append(t)
        _v.append(v)
    tokens, vars = _t, _v
    text = "".join(tokens).replace("Ġ", " ")
    i, combined = 0, []
    in_quote = False  # we just need to trace multi-word constants
    for token in text.split():
        start_i = i
        offset = len(token.replace(" ", ""))
        while offset > 0:
            offset -= len(tokens[i].strip("Ġ "))
            i += 1
        if offset != 0:
            log.error("------------")
            log.error(f"text: {text}")
            log.error(f"tokens: {tokens}")
            log.error(f"vars: {vars}")
            log.error(f"combined: {combined}")
            log.error(f"i: {i}     start_i: {start_i}     in_quote: {in_quote}")
            return "( bad )"
        if (
            token_type(token) == TType.concept
            and token not in ("interrogative", "imperative", "expressive", "(99", "99)")
            and not in_quote
        ):
            combined.append(f"{token}_{vars[start_i]}")
        else:
            combined.append(token)
        in_quote ^= token.count('"') % 2 == 1
    text = " ".join(combined)
    text = re.sub(r"99\)", ")", text)
    text = re.sub(r"\(99", "(", text)
    text = re.sub(r"</?graph>", "", text)
    text = add_bracket_to_single(text)
    return text


def postprocess_attvar_single(data):
    tokens, vars = data
    tokens = tokens[2:]  # skip special tokens
    vars = vars[1:]
    _t, _v = [], []
    for t, v in zip(tokens, vars):
        if t in ("<s>", "<pad>", "</s>"):
            continue
        elif t == "<lit>":
            t = ' "'
        elif t == "</lit>":
            t = '"'
        _t.append(t)
        _v.append(v)
    tokens, vars = _t, _v
    text = "".join(tokens).replace("Ġ", " ")
    i, combined = 0, []
    in_quote = False  # we just need to trace multi-word constants
    for token in text.split():
        start_i = i
        offset = len(token.replace(" ", ""))
        while offset > 0:
            offset -= len(tokens[i].strip("Ġ "))
            i += 1
        if offset != 0:
            log.error("------------")
            log.error(f"text: {text}")
            log.error(f"tokens: {tokens}")
            log.error(f"vars: {vars}")
            log.error(f"combined: {combined}")
            log.error(f"i: {i}     start_i: {start_i}     in_quote: {in_quote}")
            return "( bad )"
        if (
            token_type(token) == TType.concept
            and token not in ("interrogative", "imperative", "expressive", "(99", "99)")
            and not in_quote
        ):
            combined.append(f"{token}_{vars[start_i]}")
        else:
            combined.append(token)
        in_quote ^= token.count('"') % 2 == 1
    text = " ".join(combined)
    text = re.sub(r"99\)", ")", text)
    text = re.sub(r"\(99", "(", text)
    text = re.sub(r"</?graph>", "", text)
    text = add_bracket_to_single(text)
    return text


# From amrlib
def is_num(val):
    if val in ("nan", "infinity"):
        return False
    try:
        x = float(val)
        return True
    except ValueError:
        return False


# From amrlib
# Helper function token types
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


def fix_simple_string(text):
    # add variable mark
    # add bracket at right side if not balanced
    # remove some illegal tokens

    tokens = text.strip().split()

    if len(tokens) <= 2:
        return "( v0 / bad )"

    # handle the most outer bracket
    if tokens[0] == "(":
        tokens.pop(0)
    if tokens[-1] == ")":
        tokens.pop()

    processed = ["("]
    bracket = 0
    prev_is_rel = True
    in_constant = False
    var_id = 0
    for tok in tokens:
        if in_constant:
            processed.append(tok)
            if tok[-1] == '"':
                in_constant = False
        # remove illegal ( x :ARG0 :ARG1_illegal y )
        elif prev_is_rel and tok[0] == ":":
            continue
        elif tok == ")":
            # remove illegal ( x :ARG0_illegal )
            if prev_is_rel:
                processed.pop()
                need_rel, prev_is_rel = True, False
            if bracket > 0:
                processed.append(tok)
                bracket -= 1
        # remove illegal ( x :ARG0 y illegal_z illegal_p :ARG1 b )
        elif not prev_is_rel and tok[0] != ":":
            continue
        elif tok == "(":
            processed.append(tok)
            prev_is_rel = True
            bracket += 1
        elif tok[0] == '"':
            processed.append(tok)
            prev_is_rel = False
            if tok[-1] != '"':
                in_constant = True
        elif tok[0] == ":":
            processed.append(tok)
            prev_is_rel = True
        else:
            processed.extend([f"v{var_id}", "/", tok])
            var_id += 1
            prev_is_rel = False

    if bracket > 0:
        processed.extend([")"] * bracket)

    processed.append(")")
    return " ".join(processed)
