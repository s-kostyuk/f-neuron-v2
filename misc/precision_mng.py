import contextlib
from typing import Optional

import torch.backends.cuda
import torch.backends.cudnn


@contextlib.contextmanager
def precision_mng(name: Optional[str]):
    prev_dtype = torch.get_default_dtype()
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_fp16_rpr = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)

    # print(prev_dtype, prev_matmul_tf32, prev_cudnn_tf32, prev_fp16_rpr)

    if name is None:
        # Default as of torch 2.0
        new_dtype = torch.float32
        new_matmul_tf32 = False
        new_cudnn_tf32 = True
        new_fp16_rpr = True
    elif name == "fp16" or name == "float16":
        new_dtype = torch.float16
        new_matmul_tf32 = False
        new_cudnn_tf32 = False
        new_fp16_rpr = False
    elif name == "fp16rpr" or name == "float16rpr":
        new_dtype = torch.float16
        new_matmul_tf32 = False
        new_cudnn_tf32 = False
        new_fp16_rpr = True
    elif name == "fp64" or name == "float64":
        new_dtype = torch.float64
        new_matmul_tf32 = False
        new_cudnn_tf32 = False
        new_fp16_rpr = False
    elif name == "tf32":
        new_dtype = torch.float32
        new_matmul_tf32 = True
        new_cudnn_tf32 = True
        new_fp16_rpr = False
    elif name == "fp32" or name == "float32":
        new_dtype = torch.float32
        new_matmul_tf32 = False
        new_cudnn_tf32 = False
        new_fp16_rpr = False
    else:
        raise NotImplementedError(f'Precision {name} is not supported')

    torch.set_default_dtype(new_dtype)
    torch.backends.cuda.matmul.allow_tf32 = new_matmul_tf32
    torch.backends.cudnn.allow_tf32 = new_cudnn_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        new_fp16_rpr)

    try:
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            prev_fp16_rpr)
