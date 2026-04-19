import ndrs as nd
import numpy as np

def test_creation():
    t = nd.Tensor([1, 2, 3])
    assert t.shape == [3]
    assert t.dtype == nd.int32   # 自动推断为 int32

def test_add():
    a = nd.Tensor([1, 2, 3])
    b = nd.Tensor([4, 5, 6])
    c = a + b
    np.testing.assert_allclose(c.numpy(), np.array([5, 7, 9]))

def test_from_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = nd.Tensor.from_numpy(arr)
    assert t.shape == [2, 2]
    np.testing.assert_allclose(t.numpy(), arr)

def test_add_int():
    a = nd.Tensor([1,2,3], dtype=nd.int32)
    b = nd.Tensor([4,5,6], dtype=nd.int32)
    c = a + b
    assert c.dtype == nd.int32
    np.testing.assert_equal(c.numpy(), np.array([5,7,9], dtype=np.int32))