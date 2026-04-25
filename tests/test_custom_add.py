import ndrs
import numpy as np
import ctypes

def test_override_add():
    a = ndrs.Tensor([1.0, 2.0, 3.0])
    b = ndrs.Tensor([4.0, 5.0, 6.0])
    
    # 自定义加法
    def my_add(a_ptr, b_ptr, out_ptr, n, device_code, stream):
        arr_a = np.ctypeslib.as_array(ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
        arr_b = np.ctypeslib.as_array(ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
        arr_out = np.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
        np.add(arr_a, arr_b, out=arr_out)
        arr_out += 1.0
        return 0
    
    ndrs.register_binary_op(ndrs.DTYPE_FLOAT32, ndrs.BINARY_OP_ADD, "cpu", my_add)
    c = a + b
    np.testing.assert_allclose(c.numpy(), [6.0, 8.0, 10.0])