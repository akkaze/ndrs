import ndrs as nd
import numpy as np


def test_complex_dtype():
    # Define complex type as two float32 fields
    complex_dtype = nd.dtype.from_fields([("re", nd.float32), ("im", nd.float32)])

    # Create data using numpy with the corresponding numpy dtype
    np_dtype = complex_dtype.to_numpy_dtype()
    data = np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], dtype=np_dtype)

    t = nd.Tensor(data, dtype=complex_dtype)
    assert t.shape == [3]
    assert t.dtype == complex_dtype

    # Convert back to numpy
    arr = t.numpy()
    assert arr.dtype == np_dtype
    np.testing.assert_equal(arr["re"], [1.0, 3.0, 5.0])
    np.testing.assert_equal(arr["im"], [2.0, 4.0, 6.0])


def test_complex_2d():
    complex_dtype = nd.dtype.from_fields([("re", nd.float32), ("im", nd.float32)])
    np_dtype = complex_dtype.to_numpy_dtype()
    data = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=np_dtype)
    t = nd.Tensor(data, dtype=complex_dtype)
    assert t.shape == [2, 2]
    arr = t.numpy()
    assert arr["re"][0, 1] == 3
    assert arr["im"][1, 0] == 6


def test_complex_add():
    import ctypes

    complex_dtype = nd.dtype.from_fields([("re", nd.float32), ("im", nd.float32)])

    # Define complex addition callback
    def complex_add(a_ptr, b_ptr, out_ptr, n, device_code, stream):
        # Cast pointers to complex float arrays
        # For simplicity, assume float32 fields re and im are contiguous
        # We'll treat as two interleaved floats
        arr_a = np.ctypeslib.as_array(
            ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n * 2,)
        )
        arr_b = np.ctypeslib.as_array(
            ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n * 2,)
        )
        arr_out = np.ctypeslib.as_array(
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n * 2,)
        )
        # Add real and imaginary parts separately
        arr_out[0::2] = arr_a[0::2] + arr_b[0::2]  # re
        arr_out[1::2] = arr_a[1::2] + arr_b[1::2]  # im
        return 0

    nd.register_binary_op(complex_dtype, nd.BINARY_OP_ADD, "cpu", complex_add)

    # Create two complex tensors
    data1 = np.array([(1, 2), (3, 4)], dtype=complex_dtype.to_numpy_dtype())
    data2 = np.array([(5, 6), (7, 8)], dtype=complex_dtype.to_numpy_dtype())
    a = nd.Tensor(data1, dtype=complex_dtype)
    b = nd.Tensor(data2, dtype=complex_dtype)

    c = a + b
    arr = c.numpy()
    np.testing.assert_equal(arr["re"], [6, 10])
    np.testing.assert_equal(arr["im"], [8, 12])
