import ndrs as nd
import numpy as np
import pytest

def _cuda_available():
    return nd.is_cuda_available()

def _skip_if_no_cuda():
    if not _cuda_available():
        pytest.skip("CUDA not available")

class TestDevice:
    def test_device_context(self):
        _skip_if_no_cuda()
        with nd.Device("cuda", 0):
            current = nd.get_current_device()
            assert "cuda" in current

class TestStream:
    def test_stream_sync(self):
        _skip_if_no_cuda()
        s = nd.Stream()
        s.synchronize()

    def test_stream_context(self):
        _skip_if_no_cuda()
        with nd.Device("cuda", 0):
            s = nd.Stream()
            with s:
                a = nd.Tensor([1.0, 2.0])
                b = a + 1.0
                s.synchronize()
                np.testing.assert_allclose(b.numpy(), [2.0, 3.0])

class TestEvent:
    def test_event_create(self):
        _skip_if_no_cuda()
        e = nd.Event()
        assert not e.done  # 未记录时 done 应为 False
        e.synchronize()

    def test_event_record(self):
        _skip_if_no_cuda()
        with nd.Device("cuda", 0):
            s = nd.Stream()
            e = s.record()
            e.synchronize()
            assert e.done

class TestCudaVsNumpy:
    def test_add_2d(self):
        _skip_if_no_cuda()
        np_a = np.random.rand(3, 4).astype(np.float32)
        np_b = np.random.rand(3, 4).astype(np.float32)
        np_result = np_a + np_b

        with nd.Device("cuda", 0):
            tt_a = nd.Tensor(np_a.tolist(), dtype=nd.float32)
            tt_b = nd.Tensor(np_b.tolist(), dtype=nd.float32)
            tt_result = (tt_a + tt_b).to("cpu")

        np.testing.assert_allclose(tt_result.numpy(), np_result, rtol=1e-5)

    def test_device_transfer(self):
        _skip_if_no_cuda()
        np_a = np.random.rand(10).astype(np.float32)
        t_cpu = nd.Tensor(np_a.tolist(), dtype=nd.float32)
        t_gpu = t_cpu.to("cuda")
        t_back = t_gpu.to("cpu")
        np.testing.assert_allclose(t_back.numpy(), np_a)