from _ndrs import cuda as _cuda


class RawKernel:
    def __init__(self, code, name):
        self._kernel = _cuda.RawKernel(code, name)

    def __call__(self, grid, block, args, stream=None):
        self._kernel.launch(grid, block, args, stream)


class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name):
        # 构造参数列表字符串，如 "float32 a, float32 b"
        # 表达式如 "out = a + b"
        self._kernel = _cuda.ElementwiseKernel(in_params, out_params, operation, name)

    def __call__(self, *args, stream=None):
        # 分离输入和输出
        if len(args) < 2:
            raise ValueError("Need at least one input and one output")
        outputs = [args[-1]]
        inputs = args[:-1]
        self._kernel.launch(inputs, outputs, stream)
