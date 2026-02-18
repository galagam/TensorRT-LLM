# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch

_QUANT_MODE_NVFP4 = 0
_QUANT_MODE_FP8 = 1


def _trtllm_fused_add_rms_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    eps: float,
    quant_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_shape = tuple(x.shape)
    residual_shape = tuple(residual.shape)
    hidden_size = int(x_shape[-1])

    x_2d = x.reshape(-1, hidden_size).contiguous()
    residual_2d = residual.reshape(-1, hidden_size).contiguous()

    quantized, residual_out_2d, quant_scale, _ = torch.ops.trtllm.fused_add_rms_norm_quant(
        x_2d,
        residual_2d,
        weight.contiguous(),
        None if input_scale is None else input_scale.contiguous(),
        True,
        eps=eps,
        output_hp_norm=False,
        quant_mode=quant_mode,
    )

    if quant_mode == _QUANT_MODE_NVFP4:
        quantized = quantized.view(torch.uint8)
        if len(x_shape) != 2:
            quantized = quantized.reshape(*x_shape[:-1], hidden_size // 2)
    else:
        quantized = quantized.reshape(*x_shape)

    residual_out = residual_out_2d.reshape(*residual_shape)
    return quantized, residual_out, quant_scale


def trtllm_fused_add_rms_norm_nvfp4_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _trtllm_fused_add_rms_norm_quant(
        x,
        residual,
        weight,
        input_scale,
        eps,
        _QUANT_MODE_NVFP4,
    )


def trtllm_fused_add_rms_norm_fp8_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _trtllm_fused_add_rms_norm_quant(
        x,
        residual,
        weight,
        input_scale,
        eps,
        _QUANT_MODE_FP8,
    )
