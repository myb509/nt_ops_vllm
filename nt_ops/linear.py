import ninetoothed
import ninetoothed.language as ntl
import torch
import math
from typing import Callable

from nt_ops.config import (
    NT_MAX_NUM_CONFIG,
    NT_STATIC_MODE,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger
logger = init_logger(__name__)

if NT_STATIC_MODE:
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 128
else:
    BLOCK_SIZE_M = ninetoothed.block_size()
    BLOCK_SIZE_N = ninetoothed.block_size()
    BLOCK_SIZE_K = ninetoothed.block_size()


def arrangement(
    input,
    other,
    output,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    output_arranged = output.tile((block_size_m, block_size_n))

    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other = other.permute((1, 0))
    other_arranged = other.tile((block_size_k, block_size_n))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def arrangement_with_bias(
    input,
    other,
    output,
    bias,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    input_arranged, other_arranged, output_arranged = arrangement(
        input,
        other,
        output,
        block_size_m,
        block_size_n,
        block_size_k,
    )

    # bias (1, N)
    bias_arranged = bias.tile((1, block_size_n))  # (1, N/BN) x (1, BN)
    bias_arranged = bias_arranged.expand(
        (output_arranged.shape[0], -1)
    )  # (M/BM, N/BN) x (1, BN)
    return input_arranged, other_arranged, output_arranged, bias_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator


def application_with_bias(input, other, output, bias):
    application(input, other, output)
    output = output + bias


kernel = {}
kernel["no_bias"] = ninetoothed.make(
    arrangement,
    application,
    (ninetoothed.Tensor(2) for _ in range(3)),
    max_num_configs=NT_MAX_NUM_CONFIG,
)

kernel["with_bias"] = ninetoothed.make(
    arrangement_with_bias,
    application_with_bias,
    (ninetoothed.Tensor(2) for _ in range(4)),
    max_num_configs=NT_MAX_NUM_CONFIG,
)


def linear(
    input: torch.Tensor, other: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    assert input.shape[1] == other.shape[1], "Inner dimension K must match for NT GEMM"
    output_shape = (input.shape[0], other.shape[0])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if bias is not None:
        kernel["with_bias"](input, other, output, bias.view(1, -1))
    else:
        kernel["no_bias"](input, other, output)

    return output


def fake_linear(
    input: torch.Tensor, other: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.empty(
        input.shape[0], other.shape[0], dtype=input.dtype, device=input.device
    )


direct_register_custom_op("nt_linear", linear, fake_impl=fake_linear)


def nt_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.ops.vllm.nt_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    logger.info_once("\033[32mNT GEMM is enabled.\033[0m")
    return nt_unquantized_gemm
