import numpy as np
from ._ndrs import _Tensor, DTYPE_FLOAT32, DTYPE_INT32
from .dtype import get_dtype_from_id, DType
from .tensor import Tensor


def zeros(shape, dtype=None, device="cpu"):
    """创建全零张量"""
    dtype_id = get_dtype_from_id(dtype)
    if dtype_id is None:
        dtype_id = DTYPE_FLOAT32  # 默认 float32
    inner = _Tensor.zeros(shape, dtype_id, device)
    return Tensor._from_inner(inner)


def ones(shape, dtype=None, device="cpu"):
    """创建全一张量"""
    dtype_id = get_dtype_from_id(dtype)
    if dtype_id is None:
        dtype_id = DTYPE_FLOAT32
    inner = _Tensor.ones(shape, dtype_id, device)
    return Tensor._from_inner(inner)


def empty(shape, dtype=None, device="cpu"):
    """创建未初始化张量（当前实现为零初始化）"""
    dtype_id = get_dtype_from_id(dtype)
    if dtype_id is None:
        dtype_id = DTYPE_FLOAT32
    inner = _Tensor.empty(shape, dtype_id, device)
    return Tensor._from_inner(inner)
