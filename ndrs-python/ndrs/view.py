class TensorView:
    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def _from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def shape(self):
        return self._inner.shape()

    @property
    def dtype(self):
        return self._inner.dtype()

    @property
    def device(self):
        return self._inner.device()

    def contiguous(self):
        from .tensor import Tensor

        return Tensor._from_inner(self._inner.contiguous())
