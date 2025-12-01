from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from vllm.utils.torch_utils import direct_register_custom_op
from typing import Tuple

class RMSWithWeight:

    def arrangement(
        input,
        weight,
        output,
        eps,
    ):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, eps

    def application(input, weight, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    def premake(ndim):
        kernel = make(
            RMSWithWeight.arrangement,
            RMSWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSNoWeight:

    def arrangement(input, output, eps):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        return input_arranged, output_arranged, eps

    def application(input, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    def premake(ndim):
        kernel = make(
            RMSNoWeight.arrangement,
            RMSNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualWithWeight:

    def arrangement(
        input,
        weight,
        output,
        residual,
        eps,
    ):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        res_arranged = residual.tile(arrange_shape)
        res_arranged = _squeeze(res_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, res_arranged, eps

    def application(input, weight, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    def premake(ndim):
        kernel = make(
            RMSResidualWithWeight.arrangement,
            RMSResidualWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualNoWeight:

    def arrangement(input, output, residual, eps):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        res_arranged = residual.tile(arrange_shape)
        res_arranged = _squeeze(res_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        return input_arranged, output_arranged, res_arranged, eps

    def application(input, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    def premake(ndim):
        kernel = make(
            RMSResidualNoWeight.arrangement,
            RMSResidualNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


kernel = {}
max_ndim = 4
kernel["with_residual_with_weight"] = {
    i: RMSResidualWithWeight.premake(i) for i in range(2, max_ndim)
}
kernel["with_residual_no_weight"] = {
    i: RMSResidualNoWeight.premake(i) for i in range(2, max_ndim)
}
kernel["no_residual_with_weight"] = {
    i: RMSWithWeight.premake(i) for i in range(2, max_ndim)
}
kernel["no_residual_no_weight"] = {
    i: RMSNoWeight.premake(i) for i in range(2, max_ndim)
}


def rms_rw(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    kernel["with_residual_with_weight"][input.ndim](
        input, weight, output, residual, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output, residual


def fake_rms_rw(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    return output, residual


def rms_r(
    input: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    kernel["with_residual_no_weight"][input.ndim](
        input, output, residual, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output, residual


def fake_rms_r(
    input: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    return output, residual


def rms_w(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    kernel["no_residual_with_weight"][input.ndim](
        input, weight, output, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output


def fake_rms_w(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    return output


def rms(
    input: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    kernel["no_residual_no_weight"][input.ndim](
        input, output, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output


def fake_rms(
    input: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    return output


direct_register_custom_op("nt_rms_rw", rms_rw, fake_impl=fake_rms_rw)
direct_register_custom_op("nt_rms_r", rms_r, fake_impl=fake_rms_r)
direct_register_custom_op("nt_rms_w", rms_w, fake_impl=fake_rms_w)
direct_register_custom_op("nt_rms", rms, fake_impl=fake_rms)


def rms(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
):

    if weight is not None and residual is not None:
        return torch.ops.vllm.nt_rms_rw(
            input, weight.view((1,) * (input.ndim - 1) + (-1,)), residual, eps
        )
    elif weight is not None and residual is None:
        return torch.ops.vllm.nt_rms_w(
            input, weight.view((1,) * (input.ndim - 1) + (-1,)), eps
        )
    elif weight is None and residual is not None:
        return torch.ops.vllm.nt_rms_r(input, residual, eps)
    else:
        return torch.ops.vllm.nt_rms(input, eps)


def rms_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if self.variance_size_override is not None:
        return self.forward_native(x, residual)
    return rms(
        input=x, weight=self.weight.data, residual=residual, eps=self.variance_epsilon
    )
