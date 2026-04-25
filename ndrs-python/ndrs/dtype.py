import numpy as np
from .register import register_dtype

# 全局字典：dtype_id -> DType 实例
_DTYPE_REGISTRY = {}


class DType:
    """Custom dtype descriptor, similar to numpy.dtype."""

    def __init__(self, name, fields, itemsize, dtype_id):
        self.name = name
        self.fields = fields  # list of (name, ndrs_dtype_id)
        self.itemsize = itemsize
        self._id = dtype_id
        _DTYPE_REGISTRY[dtype_id] = self

    @classmethod
    def from_fields(cls, fields):
        """
        fields: list of (field_name, ndrs_dtype_id) or (field_name, (ndrs_dtype_id, offset))
        For simplicity, we ignore offset and let numpy compute itemsize.
        """
        # Convert ndrs dtype ids to numpy dtypes for layout calculation
        np_fields = []
        for field in fields:
            name, dtype_id = field[0], field[1]
            if dtype_id == 1:
                np_dtype = np.float32
            elif dtype_id == 2:
                np_dtype = np.int32
            else:
                # Could be another custom dtype, but for now not supported
                raise ValueError("Nested custom dtypes not supported yet")
            np_fields.append((name, np_dtype))
        np_dtype = np.dtype(np_fields)
        itemsize = np_dtype.itemsize
        # Unique name
        name = f"custom_{id(cls)}_{len(_DTYPE_REGISTRY)}"
        dtype_id = register_dtype(name, itemsize)
        return cls(name, fields, itemsize, dtype_id)

    @property
    def id(self):
        return self._id

    def to_numpy_dtype(self):
        """Convert to numpy dtype for array construction."""
        np_fields = []
        for name, dtype_id in self.fields:
            if dtype_id == 1:
                np_dtype = np.float32
            elif dtype_id == 2:
                np_dtype = np.int32
            else:
                raise ValueError(f"Unknown ndrs dtype id {dtype_id}")
            np_fields.append((name, np_dtype))
        return np.dtype(np_fields)

    def __repr__(self):
        return f"ndrs.DType(name={self.name}, fields={self.fields}, itemsize={self.itemsize})"


def get_dtype_from_id(dtype_id):
    """Retrieve DType object from registry by id."""
    return _DTYPE_REGISTRY.get(dtype_id)
