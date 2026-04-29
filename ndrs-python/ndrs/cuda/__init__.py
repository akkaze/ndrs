from .elementwsie_kernel import ElementwiseKernel
from ndrs._ndrs import _cuda

get_device = _cuda.get_device
set_device = _cuda.set_device
Stream = _cuda.Stream
set_stream = _cuda.set_stream
get_stream = _cuda.get_stream
get_device_count = _cuda.get_device_count
