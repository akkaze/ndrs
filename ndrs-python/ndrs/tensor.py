import numpy as np
from ._ndrs import PyTensor, DTYPE_FLOAT32, DTYPE_INT32
from .dtype import get_dtype_from_id, DType


def _get_dtype_id(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, str):
        if dtype in ("float32", "f"):
            return DTYPE_FLOAT32
        elif dtype in ("int32", "i"):
            return DTYPE_INT32
        else:
            raise ValueError(f"Unknown dtype string: {dtype}")
    if isinstance(dtype, DType):
        return dtype.id
    raise TypeError(f"Unsupported dtype type: {type(dtype)}")


class Tensor:
    def __init__(self, data, dtype=None, device=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        target_dtype_id = _get_dtype_id(dtype)
        if target_dtype_id is None:
            if data.dtype.names is not None:
                raise ValueError("Structured array requires explicit dtype")
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.int32)
                target_dtype_id = DTYPE_INT32
            else:
                data = data.astype(np.float32)
                target_dtype_id = DTYPE_FLOAT32
        else:
            if target_dtype_id == DTYPE_FLOAT32:
                data = data.astype(np.float32)
            elif target_dtype_id == DTYPE_INT32:
                data = data.astype(np.int32)
            # else custom: keep as is

        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        shape = list(data.shape)
        bytes_data = data.tobytes()
        self._inner = PyTensor.from_bytes(bytes_data, shape, target_dtype_id, device)
        self._dtype_id = target_dtype_id
        if isinstance(dtype, DType):
            self._custom_dtype = dtype
        elif target_dtype_id not in (DTYPE_FLOAT32, DTYPE_INT32):
            self._custom_dtype = get_dtype_from_id(target_dtype_id)
        else:
            self._custom_dtype = None

    @classmethod
    def from_numpy(cls, array, device=None):
        return cls(array, device=device)

    @property
    def shape(self):
        return self._inner.shape()

    @property
    def dtype(self):
        if self._dtype_id == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        elif self._dtype_id == DTYPE_INT32:
            return DTYPE_INT32
        else:
            return self._custom_dtype

    @property
    def device(self):
        return self._inner.device()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor._from_inner(self._inner + other._inner)
        raise TypeError("Unsupported operand type")

    def numpy(self):
        bytes_data = self._inner.as_bytes()
        if self._dtype_id == DTYPE_FLOAT32:
            dtype = np.float32
        elif self._dtype_id == DTYPE_INT32:
            dtype = np.int32
        else:
            dtype = self._custom_dtype.to_numpy_dtype()
        arr = np.frombuffer(bytes_data, dtype=dtype)
        return arr.reshape(self.shape)

    @classmethod
    def _from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        obj._dtype_id = inner.dtype_id()
        if obj._dtype_id not in (DTYPE_FLOAT32, DTYPE_INT32):
            obj._custom_dtype = get_dtype_from_id(obj._dtype_id)
        else:
            obj._custom_dtype = None
        return obj

    def __repr__(self):
        if self._dtype_id == DTYPE_FLOAT32:
            dtype_name = "float32"
        elif self._dtype_id == DTYPE_INT32:
            dtype_name = "int32"
        else:
            dtype_name = "custom"
        return f"Tensor(shape={self.shape}, dtype={dtype_name}, device={self.device})"
