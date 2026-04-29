# tests/test_elementwise_kernel.py
import ndrs as nd
import numpy as np
import pytest


def is_cuda_available():
    try:
        return nd.cuda.get_device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_elementwise_kernel_add_f32():
    kernel = nd.cuda.ElementwiseKernel(
        "X x, Y y",
        "Z z",
        """
        X a = x - 1;
        Y b = y * 2;
        z = (a - b) * (a - b)
        """,
        "squared_diff_super_generic",
    )
    a = nd.Tensor([1.0, 2.0, 3.0], dtype=nd.float32, device="cuda:0")
    b = nd.Tensor([4.0, 5.0, 6.0], dtype=nd.float32, device="cuda:0")
    c = kernel(a, b)
    np.testing.assert_allclose(c.contiguous().numpy(), np.array([64.0, 81.0, 100.0]))
