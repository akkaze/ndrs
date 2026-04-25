# ndrs/register.py
import ctypes
from ._ndrs import (
    register_dtype_py as _register_dtype,
    register_binary_op_raw as _register_binary_op_raw,
    DTYPE_FLOAT32,
    DTYPE_INT32,
    BINARY_OP_ADD,
)

def register_dtype(name: str, size: int) -> int:
    """Register a new custom data type, return dtype ID."""
    return _register_dtype(name, size)


def register_binary_op(dtype, kind, device, func):
    """Register binary op for built-in dtype id or DType instance."""
    from .dtype import DType
    if isinstance(dtype, DType):
        dtype_id = dtype.id
    elif isinstance(dtype, int):
        dtype_id = dtype
    else:
        raise TypeError("dtype must be DType instance or int id")
    _register_binary_op_raw(dtype_id, kind, device, _make_callback(func))


def _make_callback(func):
    CFUNC = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )

    @CFUNC
    def wrapper(a, b, out, n, device_code, stream_ptr):
        return func(a, b, out, n, device_code, stream_ptr)

    return ctypes.cast(wrapper, ctypes.c_void_p).value


__all__ = ["register_dtype", "register_binary_op"]
