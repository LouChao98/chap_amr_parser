# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by louchao


"""Masks for Transformer Grammars models."""

import dataclasses
from typing import Dict

import numpy as np
import torch
from transformers import RobertaTokenizer

from src.models.components.masking import utils as masking_utils
from src.models.components.masking.bottom_up_pointing import CloseOnlyPointer
from src.models.components.masking.masking_types import Chunk
from src.models.components.masking.single_closing_nt import (
    SingleClosingNT,
    SingleClosingNTComposingOnly,
    SingleClosingNTDiffHeads,
    SingleClosingNTOpenHead,
    SingleClosingNTStackingOnly,
)
from src.utils.pylogger import get_pylogger

from . import constants as mc
from . import cpp_masking as mcpp

log = get_pylogger(__name__)


@dataclasses.dataclass(frozen=True)
class TokenTypeRanges:
    """Mapping between token IDs ranges to token types."""

    start_token: int
    pad_token: int
    opening_non_terminal: int
    closing_non_terminal: int

    def token_type_from_token(self, seq, *, use_pytorch=False):
        """Returns an array of token types from an array of token IDs."""
        if use_pytorch:
            np_ = torch
            dtype = torch.int32
        else:
            np_ = np
            dtype = np.int32

        start_token_mask = seq == self.start_token
        pad_token_mask = seq == self.pad_token
        opening_nt_mask = seq == self.opening_non_terminal
        closing_nt_mask = seq == self.closing_non_terminal

        result = np_.full_like(seq, mc.TERMINAL, dtype=dtype)
        result[start_token_mask] = mc.SOS
        result[pad_token_mask] = mc.PAD
        result[opening_nt_mask] = mc.OPENING_NT
        result[closing_nt_mask] = mc.CLOSING_NT
        return result


def get_masking_rules(name, **kwargs):
    """Returns the masking rules instance."""
    log.info("Creating masking rules %s with kwargs=%s", name, repr(kwargs))
    if name == "stack_compose_double_closing_nt":
        # kwargs:
        #   int sequence_length
        #   int memory_length
        #   float transparency_prob
        #   bool use_relative_positions
        #   bool gather_into_new_memory: smart memory
        #   int transparency_depth_threshold:
        #           Depth below or at which the node is transparent
        #           -1 means that it's never transparent.
        #           <s> has depth 0, (DOC depth 1, so for the top level (S
        #           to be transparent, we need this to be set to 2
        cls = mcpp.StackComposeDoubleClosingNT
    elif name == "single_closing_nt":
        # kwargs: none
        cls = SingleClosingNT
    elif name == "single_closing_nt_open_head":
        # kwargs: none
        cls = SingleClosingNTOpenHead
    elif name == "single_closing_nt_diff_heads":
        # kwargs: none
        cls = SingleClosingNTDiffHeads
    elif name == "single_closing_nt_composing_only":
        # kwargs: none
        cls = SingleClosingNTComposingOnly
    elif name == "single_closing_nt_stacking_only":
        # kwargs: none
        cls = SingleClosingNTStackingOnly
    elif name == "closing_only_pointer":
        cls = CloseOnlyPointer
    elif name == "txl":
        # kwargs:
        #   int sequence_len
        #   int memory_len
        cls = mcpp.TXLCausalMasking
    else:
        raise NotImplementedError
    if kwargs is None:
        kwargs = dict()
    maskrules = cls(**kwargs)
    return maskrules


def compute_token_types(inp: Dict[str, np.ndarray], ranges: masking_utils.TokenTypeRanges) -> Dict[str, np.ndarray]:
    """Computes token types using a dictionary."""
    for key in ("inputs", "labels"):
        if ranges is not None:
            # Only ever happens for terminals on PTB
            # For CC, we have explicit ranges available, for datasets tokenised with
            # SentencePiece, we derive ranges from the .vocab file, so this is very
            # much a corner case.
            inp[f"{key}_ttypes"] = ranges.token_type_from_token(inp[key])
        else:
            inp[f"{key}_ttypes"] = np.zeros_like(inp[key])
    return inp


class MaskAgent:
    def __init__(self, tokenizer, **kwargs):
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        self.ranges = self.load_ranges(tokenizer)
        self.maskrules = get_masking_rules(**kwargs)
        self.maskrules_name = kwargs["name"]

        self.is_tg = kwargs["name"] != "txl"
        self.is_double_closing = kwargs["name"] == "stack_compose_double_closing_nt"

    def __call__(self, inputs: np.ndarray, labels: np.ndarray) -> Chunk:
        inputs_ttypes = self.ranges.token_type_from_token(inputs)
        labels_ttypes = self.ranges.token_type_from_token(labels)
        try:
            chunks = list(
                self.maskrules.chunks_for_sequence(
                    inputs,
                    inputs_ttypes,
                    # maskrules cpp ext internally use 0 as pad
                    labels + 1,
                    labels_ttypes,
                )
            )
        except RuntimeError as err:
            print(self.tokenizer.convert_ids_to_tokens(inputs))
            print(sum(t == mc.OPENING_NT for t in inputs_ttypes))
            print(sum(t == mc.CLOSING_NT for t in inputs_ttypes))
            raise err

        # We only use the first chunk and do not check the number of chunks
        # standard amr is rare to reach the max seq len limitation.

        chunk = Chunk(0, *chunks[0])
        labels = chunk.labels
        labels -= 1
        labels[chunk.labels == -1] = -100
        return chunk

    def load_ranges(self, tokenizer):
        if isinstance(tokenizer, RobertaTokenizer):
            opening = tokenizer.convert_tokens_to_ids("(99")
            closing = tokenizer.convert_tokens_to_ids("99)")
        else:
            opening = tokenizer.convert_tokens_to_ids(" (99")
            closing = tokenizer.convert_tokens_to_ids(" 99)")
        assert opening == len(tokenizer) - 2
        assert closing == len(tokenizer) - 1
        return masking_utils.TokenTypeRanges(
            start_token=tokenizer.bos_token_id,
            pad_token=self.pad_id,
            opening_non_terminal=opening,
            closing_non_terminal=closing,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["maskrules"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.maskrules = get_masking_rules(**self.kwargs)
