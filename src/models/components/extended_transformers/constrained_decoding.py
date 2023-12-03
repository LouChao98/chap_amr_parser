import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessor

import src.models.components.masking.constants as mc
from src.models.components.masking.utils import TokenTypeRanges

DEBUG = False


class TransformerGrammarDoubleLogitsProcessor(LogitsProcessor):
    def __init__(self, ranges: TokenTypeRanges) -> None:
        super().__init__()
        self.ranges = ranges

        self.prev_is_compose_closing = None  # batch size can not be determined. init later
        self.num_opening = None
        self.batch_size = None

        if DEBUG:
            self.history = None

    def _init(self, batch_size):
        self.prev_is_compose_closing = [False for _ in range(batch_size)]
        self.num_opening = [0 for _ in range(batch_size)]
        self.batch_size = batch_size
        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prev_is_compose_closing is None:
            self._init(len(input_ids))

        new_tokens = input_ids[:, -1].cpu().numpy()  # TODO combine this with TransformerGrammarMaskRules
        ttypes = self.ranges.token_type_from_token(new_tokens)

        force_closing = torch.zeros(self.batch_size, dtype=torch.bool)
        force_not_closing = torch.zeros(self.batch_size, dtype=torch.bool)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.num_opening[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                if self.prev_is_compose_closing[bidx]:
                    # now is STACKING
                    self.num_opening[bidx] -= 1
                    self.prev_is_compose_closing[bidx] = False
                else:
                    force_closing[bidx] = 1
                    self.prev_is_compose_closing[bidx] = True

            if self.num_opening[bidx] <= 0:
                force_not_closing[bidx] = 1

        neginf = float("-inf")
        scores[force_not_closing, self.ranges.closing_non_terminal] = neginf
        scores[force_closing] = neginf
        # we are not interested in these labels. we just copy the prediction of previous step.
        scores[force_closing, self.ranges.closing_non_terminal] = 0.0
        return scores

    def reorder(self, beam_idx):
        self.num_opening = [self.num_opening[i] for i in beam_idx]
        self.prev_is_compose_closing = [self.prev_is_compose_closing[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarDoubleMaskRules:
    # python reimpl

    def __init__(self, batch_size, ranges: TokenTypeRanges, mode=1, compute_relative_distance=False) -> None:
        # mode=1, use composing, as TG
        # mode=2, use stacking
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.mode = mode
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.prev_is_compose_closing = [False for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.opening_positions[bidx].append(self.length - 1)
                self.generate_stacking_att_mask(bidx, masks)

                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                if self.prev_is_compose_closing[bidx]:
                    # now is STACKING
                    self.generate_stacking_att_mask(bidx, masks)
                    masked_pos = self.length - 1 if self.mode == 1 else self.length - 2
                    self.composed_positions[bidx].append(masked_pos)
                    self.prev_is_compose_closing[bidx] = False

                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    opening = self.opening_positions[bidx].pop()
                    self.generate_compose_att_mask(bidx, masks, opening)
                    self.composed_positions[bidx].extend(range(opening, self.length - 1))
                    self.prev_is_compose_closing[bidx] = True

                    self.depth_count[bidx] -= 1
                    self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)

                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, relative_distance
        else:
            return masks

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def generate_compose_att_mask(self, bidx, mask, start):
        mask[bidx, :start] = 0
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.prev_is_compose_closing = [self.prev_is_compose_closing[i] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarSingleLogitsProcessor(LogitsProcessor):
    def __init__(self, ranges: TokenTypeRanges) -> None:
        super().__init__()
        self.ranges = ranges

        self.num_opening = None
        self.batch_size = None

        if DEBUG:
            self.history = None

    def _init(self, batch_size):
        self.num_opening = [0 for _ in range(batch_size)]
        self.batch_size = batch_size
        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.num_opening is None:
            self._init(len(input_ids))

        new_tokens = input_ids[:, -1].cpu().numpy()  # TODO combine this with TransformerGrammarMaskRules
        ttypes = self.ranges.token_type_from_token(new_tokens)

        force_not_closing = torch.zeros(self.batch_size, dtype=torch.bool)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.num_opening[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                self.num_opening[bidx] -= 1

            if self.num_opening[bidx] <= 0:
                force_not_closing[bidx] = 1

        neginf = float("-inf")
        scores[force_not_closing, self.ranges.closing_non_terminal] = neginf
        return scores

    def reorder(self, beam_idx):
        self.num_opening = [self.num_opening[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarSingleMaskRules:
    # python reimpl

    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.opening_positions[bidx].append(self.length - 1)
                self.generate_stacking_att_mask(bidx, masks)

                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(bidx, masks)
                opening = self.opening_positions[bidx].pop()
                self.composed_positions[bidx].extend(range(opening, self.length - 1))
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)

                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, relative_distance
        else:
            return masks

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarSingleOpenHeadMaskRules:
    # python reimpl

    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.opening_positions[bidx].append(self.length - 1)
                self.generate_stacking_att_mask(bidx, masks)

                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(bidx, masks)
                opening = self.opening_positions[bidx].pop()
                self.composed_positions[bidx].append(opening)
                self.composed_positions[bidx].extend(range(opening + 2, self.length - 1))
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)

                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, relative_distance
        else:
            return masks

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarSingleComposingOnlyMaskRules:
    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.opening_positions[bidx].append(self.length - 1)
                self.generate_stacking_att_mask(bidx, masks)

                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                opening = self.opening_positions[bidx].pop()
                self.generate_compose_att_mask(bidx, masks, opening)
                # self.composed_positions[bidx].extend(range(opening, self.length - 1))
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)

                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, relative_distance
        else:
            return masks

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def generate_compose_att_mask(self, bidx, mask, start):
        mask[bidx, :start] = 0
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarSingleStackingOnlyMaskRules:
    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.OPENING_NT:
                self.opening_positions[bidx].append(self.length - 1)
                self.generate_stacking_att_mask(bidx, masks)

                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                opening = self.opening_positions[bidx].pop()
                self.generate_stacking_att_mask(bidx, masks)
                self.composed_positions[bidx].extend(range(opening, self.length - 1))
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)

                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, relative_distance
        else:
            return masks

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def generate_compose_att_mask(self, bidx, mask, start):
        mask[bidx, :start] = 0
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]



class TransformerGrammarSingleDiffHeadsMaskRules:
    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.opening_positions = [[] for _ in range(self.batch_size)]
        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((2, self.batch_size, self.length), dtype=np.int32)

        for bidx, ttype in enumerate(ttypes):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            opening_positions = self.opening_positions[bidx]

            if ttype == mc.OPENING_NT:
                self.generate_stacking_att_mask(bidx, masks[0])
                cur = len(opening_positions) - 1
                prev = self.length
                while cur >= 0 and opening_positions[cur] + 1 == prev:
                    cur -= 1
                if cur > 0:
                    position = opening_positions[cur] - 1
                else:
                    position = 0

                self.generate_compose_att_mask(bidx, masks[1], position + 1)
                self.opening_positions[bidx].append(self.length - 1)
                self.depth[bidx].append(self.depth_count[bidx])
                self.depth_count[bidx] += 1

            elif ttype == mc.CLOSING_NT:
                opening = self.opening_positions[bidx].pop()
                self.generate_compose_att_mask(bidx, masks[1], opening)
                self.composed_positions[bidx].extend(range(opening, self.length - 1))
                self.generate_stacking_att_mask(bidx, masks[0])
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks[0])
                opening = opening_positions[-1] if len(opening_positions) > 0 else 0
                self.generate_compose_att_mask(bidx, masks[1], opening)
                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks.sum(0)[:, None] == 0] = 0
            return masks.transpose(1, 0, 2), relative_distance
        else:
            return masks.transpose(1, 0, 2)

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def generate_compose_att_mask(self, bidx, mask, start):
        mask[bidx, :start] = 0
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.opening_positions = [self.opening_positions[i][:] for i in beam_idx]
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class TransformerGrammarClosingOnlyPointerMaskRules:
    def __init__(self, batch_size, ranges: TokenTypeRanges, compute_relative_distance=False) -> None:
        self.batch_size = batch_size
        self.ranges = ranges
        self.length = 0
        self.compute_relative_distance = compute_relative_distance

        self.composed_positions = [[] for _ in range(self.batch_size)]
        self.depth = [[0] for _ in range(self.batch_size)]
        self.depth_count = [0 for _ in range(self.batch_size)]

        if DEBUG:
            self.history = [[] for _ in range(self.batch_size)]

    def step(self, new_tokens: torch.Tensor, new_pointers: torch.Tensor):
        new_tokens = new_tokens.cpu().numpy()
        new_pointers = new_pointers.cpu().numpy()

        ttypes = self.ranges.token_type_from_token(new_tokens)
        self.length += 1
        masks = np.ones((self.batch_size, self.length), dtype=np.int32)
        pointer_mask = np.ones((self.batch_size, self.length), dtype=np.int32)

        for bidx, (ttype, pointer) in enumerate(zip(ttypes, new_pointers)):
            if DEBUG:
                self.history[bidx].append(new_tokens[bidx])

            if ttype == mc.CLOSING_NT:
                self.generate_stacking_att_mask(bidx, masks)
                self.composed_positions[bidx].extend(range(pointer + 1, self.length - 1))
                self.generate_stacking_att_mask(bidx, pointer_mask)
                self.depth_count[bidx] -= 1
                self.depth[bidx].append(self.depth_count[bidx])
            else:
                self.generate_stacking_att_mask(bidx, masks)
                self.generate_stacking_att_mask(bidx, pointer_mask)
                if self.length > 1:
                    self.depth[bidx].append(self.depth_count[bidx])
                else:
                    self.depth_count[bidx] += 1

        pointer_mask[:, 0] = 0
        pointer_mask[:, -1] = 0
        # print(pointer_mask)
        if self.compute_relative_distance:
            depth = np.asarray(self.depth)
            relative_distance = depth[:, -1:, None] - depth[:, None]
            relative_distance[masks[:, None] == 0] = 0
            return masks, pointer_mask[:, None], relative_distance
        else:
            return masks, pointer_mask[:, None]

    def generate_stacking_att_mask(self, bidx, mask):
        mask[bidx, self.composed_positions[bidx]] = 0

    def reorder(self, beam_idx):
        self.composed_positions = [self.composed_positions[i][:] for i in beam_idx]
        self.depth = [self.depth[i][:] for i in beam_idx]
        self.depth_count = [self.depth_count[i] for i in beam_idx]
        if DEBUG:
            self.history = [self.history[i][:] for i in beam_idx]


class ForcedSecondTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, second_token_ids: int):
        self.second_token_ids = second_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 2:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.second_token_ids]] = -float("inf")
            scores[:, self.second_token_ids] = 0
        return scores


class ForcedNewConceptAfterOpeningLogit(LogitsProcessor):
    def __init__(self, opening, normal_tokens):
        self.not_normal_tokens = ~normal_tokens
        self.opening = opening

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len >= 2:
            mask = input_ids[:, -1] == self.opening
            mask = mask.unsqueeze(-1) & self.not_normal_tokens.unsqueeze(0)
            scores[mask] = float("-inf")
        return scores


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

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
    tg_maskrule = TransformerGrammarDoubleMaskRules(batch_size, ranges)
    tg_post_maskrule = TransformerGrammarDoubleMaskRules(batch_size, ranges, mode=2)

    example = " (99 (99 (99 a b 99) c 99) (99 d e 99) 99)"
    # example = " (99 (99 (99 (99 a b 99) c 99) d 99) e 99)"
    # example = " (99 a (99 b (99 c (99 d e 99) 99) 99) 99)"
    ids = tokenizer(example)["input_ids"]

    # gold_maskrules = masking_utils.get_masking_rules(
    #     "stack_compose_double_closing_nt", sequence_length=40, memory_length=40
    # )
    # gold_maskrules2 = masking_utils.get_masking_rules(
    #     "stack_compose_double_closing_nt", sequence_length=10, memory_length=10, use_stacking=True
    # )
    item = dict(inputs=np.array(ids), labels=np.array(ids))
    item = masking_utils.compute_token_types(item, ranges)

    # print("stack_compose_double_closing_nt")
    # chunks = list(
    #     gold_maskrules.chunks_for_sequence(
    #         item["inputs"],
    #         item["inputs_ttypes"],
    #         item["labels"],
    #         item["labels_ttypes"],
    #     )
    # )
    # print(chunks[0][0])
    # print(chunks[0][4].sum())
    # exit(0)
    # print("====")

    # print('stack_compose_double_closing_nt, using_stacking')

    # chunks2 = list(
    #     gold_maskrules2.chunks_for_sequence(
    #         item["inputs"],
    #         item["inputs_ttypes"],
    #         item["labels"],
    #         item["labels_ttypes"],
    #     )
    # )
    # print(chunks2[0][0])
    # print(chunks2[0][4])

    # print("====")

    # print('show steps')

    # for id in chunks[0][0].tolist():
    #     print(tokenizer.convert_ids_to_tokens(id))
    #     print("TG:  ", tg_maskrule.step(torch.tensor([id])).squeeze(0))
    #     print("TGP: ", tg_post_maskrule.step(torch.tensor([id])).squeeze(0))

    # # print(chunks[0][5])
    # simple_maskrules = masking_utils.get_masking_rules("single_closing_nt")
    # chunks3 = list(
    #     simple_maskrules.chunks_for_sequence(
    #         item["inputs"],
    #         item["inputs_ttypes"],
    #         item["labels"],
    #         item["labels_ttypes"],
    #     )
    # )
    # _len = chunks3[0][0].tolist().index(tokenizer.eos_token_id) + 1
    # mat = chunks3[0][4][:_len, :_len]
    # ticks = tokenizer.convert_ids_to_tokens(ids)
    # plt.matshow(mat, vmin=0, vmax=1)
    # plt.xticks(np.arange(_len), ticks, rotation=90)
    # plt.yticks(np.arange(_len), ticks)
    # fig = plt.gcf()
    # fig.set_size_inches(6, 6)
    # plt.savefig("a.png", dpi=100)
    # breakpoint()

    # print("====")

    simple_maskrules = masking_utils.get_masking_rules("single_closing_nt_stacking_only")
    chunks3 = list(
        simple_maskrules.chunks_for_sequence(
            item["inputs"],
            item["inputs_ttypes"],
            item["labels"],
            item["labels_ttypes"],
        )
    )
    _len = chunks3[0][0].tolist().index(tokenizer.eos_token_id) + 1
    mat = chunks3[0][4][:_len, :_len]
    ticks = tokenizer.convert_ids_to_tokens(ids)
    plt.matshow(mat, vmin=0, vmax=1)
    plt.xticks(np.arange(_len), ticks, rotation=90)
    plt.yticks(np.arange(_len), ticks)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("a.png", dpi=100)
    # breakpoint()
    _maskrules = TransformerGrammarSingleStackingOnlyMaskRules(batch_size, ranges)
    for id in chunks3[0][0].tolist():
        print(tokenizer.convert_ids_to_tokens(id))
        m = _maskrules.step(torch.tensor([id]))
        print(m)

    print("====")

    # print('relative distance')

    # tg_maskrule2 = TransformerGrammarDoubleMaskRules(batch_size, ranges, compute_relative_distance=True)

    # for id in chunks[0][0].tolist():
    #     print(tokenizer.convert_ids_to_tokens(id))
    #     print("TG:  ", tg_maskrule2.step(torch.tensor([id]))[1][0, 0])

    # print(chunks[0][5][:, 10:])
    # # breakpoint()

    # print("====")

    # print("single")

    # single_maskrules = masking_utils.get_masking_rules("single_closing_nt")
    # single_maskrules_incr = TransformerGrammarSingleMaskRules(batch_size, ranges, compute_relative_distance=True)
    # chunks = list(
    #     single_maskrules.chunks_for_sequence(
    #         item["inputs"],
    #         item["inputs_ttypes"],
    #         item["labels"],
    #         item["labels_ttypes"],
    #     )
    # )
    # print(chunks[0][0])
    # print(chunks[0][4])
    # print(chunks[0][5])

    # for id in chunks[0][0].tolist():
    #     print(tokenizer.convert_ids_to_tokens(id))
    #     m, r = single_maskrules_incr.step(torch.tensor([id]))
    #     print(m)
    #     print(r[0, 0])

    # breakpoint()

    # print("====")

    # print("single")

    # single_maskrules = masking_utils.get_masking_rules("single_closing_nt_diff_heads")
    # single_maskrules_incr = TransformerGrammarSingleDiffHeadsMaskRules(
    #     batch_size, ranges, compute_relative_distance=True
    # )
    # chunks = list(
    #     single_maskrules.chunks_for_sequence(
    #         item["inputs"],
    #         item["inputs_ttypes"],
    #         item["labels"],
    #         item["labels_ttypes"],
    #     )
    # )
    # print(chunks[0][0])
    # print(chunks[0][4].shape)
    # print(chunks[0][4][0])
    # print(chunks[0][4][1])
    # # print(chunks[0][5])

    # _len = chunks[0][0].tolist().index(tokenizer.eos_token_id) + 1
    # mat = chunks[0][4].cpu().numpy()[:_len, :_len]
    # ticks = tokenizer.convert_ids_to_tokens(ids)
    # plt.matshow(mat[1], vmin=0, vmax=1)
    # plt.xticks(np.arange(_len), ticks, rotation=90)
    # plt.yticks(np.arange(_len), ticks)
    # fig = plt.gcf()
    # fig.set_size_inches(6, 6)
    # plt.savefig("a.png", dpi=100)

    # _len = chunks[0][0].tolist().index(tokenizer.eos_token_id) + 1
    # mat = chunks[0][4].cpu().numpy()[:_len, :_len]
    # ticks = tokenizer.convert_ids_to_tokens(ids)
    # plt.matshow(mat[0], vmin=0, vmax=1)
    # plt.xticks(np.arange(_len), ticks, rotation=90)
    # plt.yticks(np.arange(_len), ticks)
    # fig = plt.gcf()
    # fig.set_size_inches(6, 6)
    # plt.savefig("b.png", dpi=100)

    # ids = tokenizer(
    #     [
    #         " (99 a (99 b 99) c 99) d e",
    #         " (99 (99 a 99) b (99 c 99) 99)",
    #     ]
    # )["input_ids"]
    # single_maskrules_incr = TransformerGrammarSingleDiffHeadsMaskRules(2, ranges, compute_relative_distance=True)

    # buffer = []
    # for _offset, id in enumerate(zip(*ids)):

    #     m, r = single_maskrules_incr.step(torch.tensor(id))
    #     buffer.append(m[1])
    #     # _l = m.shape[-1]
    #     # m = torch.from_numpy(m)
    #     # assert (m[0] == chunks[0][4][:, _offset, :_l]).all()
    #     # print(r[0, 0])

    # for item in buffer:
    #     print(item[0])
    # for item in buffer:
    #     print(item[1])
