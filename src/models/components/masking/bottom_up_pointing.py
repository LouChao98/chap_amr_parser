import torch
from torch.nn.utils.rnn import pad_sequence

import src.models.components.masking.constants as mc


class CloseOnlyPointer:
    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        # if pointer=i, a new span starts AFTER i.

        masks = []
        processed_inputs = 0
        pointers = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((processed_inputs + 1,), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                opening_positions.append(processed_inputs)
                depth_count += 1

            elif ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(mask, composed_positions)
                opening = opening_positions.pop()
                composed_positions.extend(range(opening, processed_inputs))
                depth_count -= 1
                processed_inputs += 1
                pointers.append(opening - 1)
                depth.append(depth_count)
                masks.append(mask)

            else:
                self.generate_stacking_att_mask(mask, composed_positions)
                masks.append(mask)
                pointers.append(-100)
                processed_inputs += 1
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

        attn_mask = pad_sequence(masks, True, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask == 0] = 0
        pointers = torch.tensor(pointers[1:] + [-100])
        pointer_mask = attn_mask.tril(-2)
        pointer_mask[:-1] = pointer_mask[1:].clone()
        pointer_mask[:, 0] = 0

        # sanity check
        m = pointers >= 0
        _p = pointers[m]
        _m = pointer_mask[m]
        assert _m.gather(1, _p.unsqueeze(1)).sum() == m.sum()

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)
        return [
            (
                torch.tensor([i for i, t in zip(inputs, inputs_ttypes) if t != mc.OPENING_NT]),
                None,
                torch.tensor([i for i, t in zip(labels, labels_ttype) if t != mc.OPENING_NT]),
                None,
                attn_mask,
                attn_relpos,
                None,
                None,
                None,
                None,
                depth,
                None,
                None,
                None,
                None,
                pointers,
                pointer_mask,  # mask out impossible positions
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer

    from src.models.components.extended_transformers.constrained_decoding import (
        TransformerGrammarClosingOnlyPointerMaskRules,
    )

    from ..masking import utils as masking_utils

    batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained("data/AMR/bart/model-base-amrtoken")
    opening = tokenizer.convert_tokens_to_ids(" (99")
    closing = tokenizer.convert_tokens_to_ids(" 99)")
    ranges = masking_utils.TokenTypeRanges(
        start_token=tokenizer.bos_token_id,
        pad_token=tokenizer.pad_token_id,
        opening_non_terminal=opening,
        closing_non_terminal=closing,
    )

    example = " (99 (99 a a 99) b (99 c 99) 99)"
    ids = tokenizer(example)["input_ids"]
    maskrules = masking_utils.get_masking_rules("closing_only_pointer")
    item = dict(inputs=np.array([0] + ids[:-1]), labels=np.array(ids))
    item = masking_utils.compute_token_types(item, ranges)

    chunks = list(
        maskrules.chunks_for_sequence(
            item["inputs"],
            item["inputs_ttypes"],
            item["labels"],
            item["labels_ttypes"],
        )
    )

    _len = chunks[0][2].tolist().index(tokenizer.eos_token_id) + 1
    mat = chunks[0][-1].cpu().numpy()[:_len, :_len]
    ticks = tokenizer.convert_ids_to_tokens(chunks[0][0])
    plt.matshow(mat, vmin=0, vmax=1)
    plt.xticks(np.arange(_len), ticks, rotation=90)
    plt.yticks(np.arange(_len), ticks)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("a.png", dpi=100)

    _len = chunks[0][2].tolist().index(tokenizer.eos_token_id) + 1
    mat = chunks[0][4].cpu().numpy()[:_len, :_len]
    ticks = tokenizer.convert_ids_to_tokens(chunks[0][0])
    plt.matshow(mat, vmin=0, vmax=1)
    plt.xticks(np.arange(_len), ticks, rotation=90)
    plt.yticks(np.arange(_len), ticks)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("b.png", dpi=100)

    print(chunks[0][0])
    print(chunks[0][2])
    print(chunks[0][-2])

    incr_maskrules = TransformerGrammarClosingOnlyPointerMaskRules(1, ranges, False)

    for id, p in zip(chunks[0][0].tolist(), [0] + chunks[0][-2].tolist()):
        print(tokenizer.convert_ids_to_tokens(id), id, p)
        output = incr_maskrules.step(torch.tensor([id]), torch.tensor([p]))
        print("TG:  ", output[0])
        print("PM:  ", output[1])
