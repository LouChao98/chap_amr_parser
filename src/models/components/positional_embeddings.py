# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class AliBi(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def forward(self, x, relative_distance):
        relative_distance = -relative_distance.abs()
        slopes = self.slopes.to(x.dtype)
        a = relative_distance[:, None] * slopes[None, :, None, None]

        return x + a
