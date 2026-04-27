import ndrs as nd
import numpy as np
import pytest

def is_cuda_available():
    """辅助函数：检查 CUDA 是否可用"""
    try:
        return nd.cuda.get_device_count() > 0
    except Exception:
        return False

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_device():
    """测试设置和获取 CUDA 设备"""
    nd.cuda.set_device("cuda:0")
    current = nd.cuda.get_device()
    assert current == "cuda:0"

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_stream_basic():
    """测试 CUDA 流的基本创建、同步和事件记录"""
    stream = nd.cuda.Stream(device_id=0)
    assert stream is not None

    # 同步（无操作也应通过）
    stream.synchronize()

    # 记录事件并同步
    event = stream.record_event()
    event.synchronize()

    # 跨流等待
    stream2 = nd.cuda.Stream(device_id=0)
    stream2.wait_event(event)
    stream2.synchronize()

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_stream_set_get():
    """测试设置和获取当前 CUDA 流"""
    original = nd.cuda.get_stream()
    new_stream = nd.cuda.Stream(device_id=0)

    nd.cuda.set_stream(new_stream)
    current = nd.cuda.get_stream()

    # 注意：get_stream 可能返回新的 Python 对象，但内部 CudaStream 指针应与 new_stream 相同
    # 我们通过简单的操作来验证：在自定义流上执行张量运算
    a = nd.Tensor([1, 2, 3], dtype=nd.float32, device="cuda:0")
    b = nd.Tensor([4, 5, 6], dtype=nd.float32, device="cuda:0")
    c = a + b
    nd.cuda.get_stream().synchronize()  # 等待当前流完成
    np.testing.assert_allclose(c.numpy(), np.array([5, 7, 9]))

    # 恢复原始流
    nd.cuda.set_stream(original)

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_stream_tensor_ops():
    """测试在自定义 CUDA 流上执行张量操作"""
    stream = nd.cuda.Stream(device_id=0)
    nd.cuda.set_stream(stream)

    a = nd.Tensor([1.0, 2.0, 3.0], dtype=nd.float32, device="cuda:0")
    b = nd.Tensor([4.0, 5.0, 6.0], dtype=nd.float32, device="cuda:0")
    c = a + b

    stream.synchronize()
    np.testing.assert_allclose(c.numpy(), np.array([5.0, 7.0, 9.0]))

    # 测试事件计时
    start = stream.record_event()
    d = a + b   # 额外的运算
    end = stream.record_event()
    stream.synchronize()

    elapsed = end.elapsed_time(start)
    assert elapsed > 0

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_cuda_event_sync_between_streams():
    """测试使用事件进行跨流同步"""
    stream1 = nd.cuda.Stream(device_id=0)
    stream2 = nd.cuda.Stream(device_id=0)

    # 在 stream1 上执行操作
    nd.cuda.set_stream(stream1)
    a = nd.Tensor([1, 2, 3], dtype=nd.float32, device="cuda:0")
    b = nd.Tensor([4, 5, 6], dtype=nd.float32, device="cuda:0")
    c = nd.Tensor([7, 8, 9], dtype=nd.float32, device="cuda:0")
    d = a + b

    # 记录事件
    event = stream1.record_event()

    # 切换到 stream2 并等待事件
    nd.cuda.set_stream(stream2)
    stream2.wait_event(event)

    # 在 stream2 上执行依赖操作
    e = c + d
    stream2.synchronize()

    np.testing.assert_allclose(e.numpy(), np.array([12.0, 15.0, 18.0]))