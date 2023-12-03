import torch
from torch.nn.utils.rnn import pad_sequence

import src.models.components.masking.constants as mc


class SingleClosingNT:
    def __init__(self):
        pass

    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        masks = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((length,), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                opening_positions.append(length - 1)
                self.generate_stacking_att_mask(mask, composed_positions)
                depth.append(depth_count)
                depth_count += 1
            elif ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(mask, composed_positions)
                opening = opening_positions.pop()
                composed_positions.extend(range(opening, length - 1))
                depth_count -= 1
                depth.append(depth_count)
            else:
                self.generate_stacking_att_mask(mask, composed_positions)
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

            masks.append(mask)

        attn_mask = pad_sequence(masks, True, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask == 0] = 0

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)

        return [
            (
                inputs,
                inputs_ttypes,
                labels,
                labels_ttype,
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
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0


class SingleClosingNTOpenHead:
    def __init__(self):
        pass

    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        masks = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((length,), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                opening_positions.append(length - 1)
                self.generate_stacking_att_mask(mask, composed_positions)
                depth.append(depth_count)
                depth_count += 1
            elif ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(mask, composed_positions)
                opening = opening_positions.pop()
                composed_positions.append(opening)
                composed_positions.extend(range(opening + 2, length - 1))
                depth_count -= 1
                depth.append(depth_count)
            else:
                self.generate_stacking_att_mask(mask, composed_positions)
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

            masks.append(mask)

        attn_mask = pad_sequence(masks, True, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask == 0] = 0

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)

        return [
            (
                inputs,
                inputs_ttypes,
                labels,
                labels_ttype,
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
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0


class SingleClosingNTComposingOnly:
    def __init__(self):
        pass

    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        masks = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((length,), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                opening_positions.append(length - 1)
                self.generate_stacking_att_mask(mask, composed_positions)
                depth.append(depth_count)
                depth_count += 1
            elif ttype == mc.CLOSING_NT:
                opening = opening_positions.pop()
                self.generate_compose_att_mask(mask, opening, composed_positions)
                # composed_positions.extend(range(opening, length - 1))
                depth_count -= 1
                depth.append(depth_count)
            else:
                self.generate_stacking_att_mask(mask, composed_positions)
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

            masks.append(mask)

        attn_mask = pad_sequence(masks, True, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask == 0] = 0

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)

        return [
            (
                inputs,
                inputs_ttypes,
                labels,
                labels_ttype,
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
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0


class SingleClosingNTStackingOnly:
    def __init__(self):
        pass

    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        masks = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((length,), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                opening_positions.append(length - 1)
                self.generate_stacking_att_mask(mask, composed_positions)
                depth.append(depth_count)
                depth_count += 1
            elif ttype == mc.CLOSING_NT:
                opening = opening_positions.pop()
                self.generate_stacking_att_mask(mask, composed_positions)
                composed_positions.extend(range(opening, length - 1))
                depth_count -= 1
                depth.append(depth_count)
            else:
                self.generate_stacking_att_mask(mask, composed_positions)
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

            masks.append(mask)

        attn_mask = pad_sequence(masks, True, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask == 0] = 0

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)

        return [
            (
                inputs,
                inputs_ttypes,
                labels,
                labels_ttype,
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
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0


class SingleClosingNTDiffHeads:
    # for closing:
    #   1 for composing, 0 for stacking
    # for others:
    #   if is first (opening), see stack elements under the lowest non-empty parents
    #   if is not first, composing

    def __init__(self):
        pass

    def chunks_for_sequence(self, inputs, inputs_ttypes, labels, labels_ttype):
        masks = []
        opening_positions = []
        composed_positions = []
        depth = [0]
        depth_count = 0

        for length, ttype in enumerate(inputs_ttypes, start=1):
            mask = torch.ones((2, length), dtype=torch.long)

            if ttype == mc.OPENING_NT:
                self.generate_stacking_att_mask(mask[0], composed_positions)
                cur = len(opening_positions) - 1
                prev = length
                while cur >= 0 and opening_positions[cur] + 1 == prev:
                    cur -= 1
                if cur > 0:
                    position = opening_positions[cur] - 1
                else:
                    position = 0

                self.generate_compose_att_mask(mask[1], position + 1, composed_positions)
                opening_positions.append(length - 1)
                depth.append(depth_count)
                depth_count += 1
            elif ttype == mc.CLOSING_NT:
                opening = opening_positions.pop()
                self.generate_compose_att_mask(mask[1], opening, composed_positions)
                composed_positions.extend(range(opening, length - 1))
                self.generate_stacking_att_mask(mask[0], composed_positions)
                depth_count -= 1
                depth.append(depth_count)
            else:
                self.generate_stacking_att_mask(mask[0], composed_positions)
                opening = opening_positions[-1] if len(opening_positions) > 0 else 0
                self.generate_compose_att_mask(mask[1], opening, composed_positions)
                if length > 1:
                    depth.append(depth_count)
                else:
                    depth_count += 1

            masks.append(mask.T)

        attn_mask = pad_sequence(masks, True, 0).movedim(2, 0)
        depth = torch.tensor(depth)
        attn_relpos = depth[:, None] - depth[None, :]
        attn_relpos[attn_mask.sum(0) == 0] = 0

        # for compatible
        attn_relpos = torch.cat([torch.zeros_like(attn_relpos), attn_relpos], dim=1)

        return [
            (
                inputs,
                inputs_ttypes,
                labels,
                labels_ttype,
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
            )
        ]

    def generate_stacking_att_mask(self, mask, composed_positions):
        mask[composed_positions] = 0

    def generate_compose_att_mask(self, mask, start, composed_positions):
        mask[:start] = 0
        mask[composed_positions] = 0
