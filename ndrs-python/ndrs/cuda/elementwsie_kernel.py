from ndrs._ndrs import _cuda
from ..tensor import Tensor
from ..view import TensorView
from ..creation import empty

# ndrs/python/ndrs/cuda.py 中的 ElementwiseKernel 类


class ElementwiseKernel:
    def __init__(self, in_params, out_param, operation, name=None):
        if name is None:
            name = "elementwise_kernel"
        # 处理 in_params：字符串或列表
        if isinstance(in_params, str):
            in_params_str = in_params
        else:
            in_params_str = ", ".join(in_params)
        # 完整参数字符串，如 "X x, Y y, Z out"
        params_str = f"{in_params_str}, {out_param}"
        self._kernel = _cuda._ElementwiseKernel(params_str, operation, name)
        self._in_params = [p.strip().split()[-1] for p in in_params_str.split(",")]
        self._out_param = out_param.split()[-1]

    def __call__(self, *inputs, out=None, stream=None):
        if len(inputs) != len(self._in_params):
            raise ValueError(
                f"Expected {len(self._in_params)} inputs, got {len(inputs)}"
            )
        # 转换为视图
        input_views = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                input_views.append(inp.as_view()._inner)
            elif isinstance(inp, TensorView):
                input_views.append(inp._inner)
            else:
                raise TypeError("Inputs must be Tensor or TensorView")
        # 输出视图
        if out is None:
            # 自动创建
            shape = input_views[0].shape()
            dtype = input_views[0].dtype()
            device = input_views[0].device()
            from ..creation import empty

            out_tensor = empty(shape, dtype, device)
            out_view = out_tensor.as_view()._inner
        else:
            if isinstance(out, Tensor):
                out_view = out.as_view()._inner
            elif isinstance(out, TensorView):
                out_view = out._inner
            else:
                raise TypeError("out must be Tensor or TensorView")
        # 调用底层内核（Rust 方法签名：launch(self, inputs, output, stream)）
        self._kernel.launch(input_views, out_view, stream)
        return out if out is not None else TensorView._from_inner(out_view)
