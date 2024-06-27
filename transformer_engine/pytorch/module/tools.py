import os
import torch

from ..utils import (
    cast_if_needed,
    assert_dim_for_fp8_exec,
)
from ..cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    cast_to_fp8,
)
from .base import (
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
# from ..constants import GemmParallelModes, dist_group_type
# from ..jit import no_torch_dynamo
# from ..graph import is_graph_capturing
from ..float8_tensor import Float8Tensor

_FP8_DEBUG = int(os.getenv("FP8_DEBUG", "0"))

def linear_bf16_forward(
    weight,
    bias,
    use_bias,
    activation_dtype,
    inputmat_total
    ):


    if _FP8_DEBUG:
        print('[Linear]: bf16 forward')

    weight = cast_if_needed(weight, activation_dtype)
    bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

    dim_size = list(inputmat_total.size())
    dim_size[1] = weight.size(0)
    out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

    _ = gemm(
        weight,
        inputmat_total,
        activation_dtype,
        get_workspace(),
        bias=bias,
        use_bias=use_bias,
        out=out,
        ub_algo=None,
        ub=None,
        extra_output_tensor=None,
    )

    return out

def linear_fp8_forward(
    fp8_meta,
    is_grad_enabled,
    weight: torch.Tensor,  # w in bf16, shape: [out, in]
    sequence_parallel,
    inputmat,              # x in bf16, shape: [-1, in]
    update_fp8_weights,
    weight_fp8: Float8Tensor,
    weight_t_fp8,
    activation_dtype,
    bias,
    use_bias,
    ):

    if _NVTE_DEBUG:
        print('[JQ] linear fp8 forward')

    assert_dim_for_fp8_exec(inputmat)
    assert_dim_for_fp8_exec(weight)

    # INPUT fp8 --------------------------------------------------------------
    # IN: inputmat (tensor)
    # OUT: inputmat_tensor_in_fp8, inputmat_t_tensor_in_fp8
    fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
    assert not isinstance(inputmat, Float8Tensor), "Error! NOT support input Float8Tensor!"

    if (
        not fp8_meta["recipe"].override_linear_precision.wgrad
        and is_grad_enabled
        and weight.requires_grad
        and not sequence_parallel
    ):
        # FP8 input for forward, FP8 input transpose for backward wgrad
        inputmat_tensor_in_fp8, inputmat_t_tensor_in_fp8 = fp8_cast_transpose_fused(
            inputmat,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )
    else:
        # FP8 input for forward
        inputmat_tensor_in_fp8 = cast_to_fp8(
            inputmat,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )
    # INPUT fp8 --------------------------------------------------------------


    # W fp8 --------------------------------------------------------------
    if update_fp8_weights:
        # Need to cast weights to FP8
        weight_fp8 = Float8Tensor(
            data=weight_fp8._data,
            fp8_meta=fp8_meta,
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
        )
        fp8_cast_transpose_fused(
            weight,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            cast_out=weight_fp8._data,
            transpose_out=weight_t_fp8._data,
            noop_flag=skip_fp8_weight_update,
        )
    # W fp8 --------------------------------------------------------------

    # Out --------------------------------------------------------------
    bias_dtype = (
        torch.bfloat16
        if activation_dtype == torch.float32
        else activation_dtype
    )
    bias = cast_if_needed(bias, bias_dtype) if use_bias else bias
    proj_out_index, meta_tensor, proj_out_tetype, proj_out_pttype = (
        None, None, None, activation_dtype)

    dim_size = list(inputmat.size())
    dim_size[1] = weight.size(0)
    out = torch.empty(dim_size, dtype=proj_out_pttype, device=inputmat.device)
    # Out --------------------------------------------------------------

    _ = fp8_gemm(
        weight_fp8._data,         # W
        fp8_meta["scaling_fwd"].scale_inv,
        tex.FP8FwdTensors.GEMM1_WEIGHT,
        fp8_dtype_forward,
        inputmat_tensor_in_fp8,   # X
        fp8_meta["scaling_fwd"].scale_inv,
        tex.FP8FwdTensors.GEMM1_INPUT,
        fp8_dtype_forward,
        proj_out_pttype,
        get_workspace(),
        bias=bias,
        use_bias=use_bias,
        use_split_accumulator=_2X_ACC_FPROP,
        out=out,   # Y tensor
        ub_algo=None,
        ub=None,
        extra_output_tensor=None,
        out_index=None, # proj_out_index,
        fp8_meta_tensor = None, #meta_tensor,
        D_dtype = None, # proj_out_tetype,
    )

    return out

def compute_similarity(a, b):
    cos_fn = torch.nn.CosineSimilarity(dim=0)
    cos = cos_fn(a, b).item()
    # TODO: normalize?
    dot = torch.dot(a, b).item()

    if _FP8_DEBUG:
        print(f'Compute similarity, cos: {cos:.3f}, dot: {dot:.3f}')

    return cos, dot
