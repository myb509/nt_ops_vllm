import ninetoothed
import ninetoothed.language as ntl
import functools
import torch
import typing
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.custom_op import CustomOp
import torch.nn.functional as F


class SiluAndMul:
    def arrangement(x, y, out):
        BLOCK_SIZE = ninetoothed.Symbol("BLOCK_SIZE", constexpr=True)
        ndim = x.ndim
        x_arranged = x.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        y_arranged = y.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        output_arranged = out.tile((1,) * (ndim - 1) + (BLOCK_SIZE,))
        return x_arranged, y_arranged, output_arranged

    def application(x, y, out):
        x_fp32 = ntl.cast(x, ntl.float32)
        y_fp32 = ntl.cast(y, ntl.float32)
        out = x_fp32 * (1 / (1 + ntl.exp(-x_fp32))) * y_fp32

    def premake(ndim: int):
        kernel = ninetoothed.make(
            SiluAndMul.arrangement,
            SiluAndMul.application,
            (
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
                ninetoothed.Tensor(ndim),
            ),
        )
        return kernel


kernel = {}
kernel["siluAndMul"] = {i: SiluAndMul.premake(i) for i in range(2, 4)}


def siluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    ndim = x.ndim
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    x_in = x[..., :d]
    y_in = x[..., d:]
    kernel["siluAndMul"][ndim](x_in, y_in, out, BLOCK_SIZE=d)
    return out


def fake_siluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op("nt_silu_and_mul", siluAndMul, fake_impl=fake_siluAndMul)


def silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.nt_silu_and_mul(x)
