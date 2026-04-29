# ndrs

[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README_zh.md)

**ndrs** 是一个类似 NumPy 的 Rust 张量库，提供多维数组（张量）运算，并支持**可选的 CUDA GPU 加速**。它强调**零拷贝视图**、高效的跨步操作和灵活的所有权模型。

---

## 📑 目录

- [特性](#-特性)
- [快速上手](#-快速上手)
  - [基本 CPU 用法](#基本-cpu-用法tensor-宏)
  - [指定 dtype/device 创建张量](#指定-dtypedevice-创建张量)
  - [使用 CUDA 流的 GPU 用法](#使用-cuda-流的-gpu-用法)
  - [NPY 文件 I/O](#npy-文件-io-numpy-兼容)
  - [自定义原始数据类型](#自定义原始数据类型)
  - [自定义逐元素内核（GPU）](#自定义逐元素内核-gpu)
  - [底层 RawKernel（动态 CUDA 内核）](#底层-rawkernel动态-cuda-内核)
- [核心概念](#-核心概念)
- [Cargo 特性](#-cargo-特性)
- [测试](#-测试)
- [Python 绑定](#-python-绑定)
  - [安装](#安装)
  - [基本用法](#基本用法)
  - [Python 中的 CUDA 流和事件](#python-中的-cuda-流和事件)
  - [自定义 dtype 和操作](#自定义-dtype-和操作)
  - [Python API 参考](#python-api-参考)
- [性能考虑](#️-性能考虑)
- [许可证](#-许可证)
- [贡献](#-贡献)
- [致谢](#-致谢)

---

## ✨ 特性

- **N 维张量** – 形状、步长和字节级数据存储。
- **基于视图的操作** – 切片、广播、转置和重塑，无需复制数据。
- **高效跨步复制** – 在非连续布局之间快速移动数据。
- **线程局部和线程安全变体**
  - `Rc<RefCell<Tensor>>` 用于单线程高性能
  - `Arc<ReentrantMutex<RefCell<Tensor>>>` 用于多线程和 Python 绑定
- **GPU 加速** – 透明的 CPU ↔ GPU 传输，用于逐元素操作的 CUDA 内核。
- **CUDA 流支持** – 异步执行，用于计时和跨流同步的事件。
- **运算符重载** – 支持广播形状的张量 `+` 和 `+=`。
- **类 Python 切片** – 直观的 `s!` 宏：`s![1..4:2, ..]`。
- **广播** – 自动形状扩展。
- **动态 CUDA 内核** – 运行时从 PTX 或 CUDA C++ 源代码编译并启动内核，使用 `RawKernel`。
- **自定义逐元素内核** – 定义自己的逐元素操作（例如 `out = a + b * c`），适用于任意秩和 dtype 的张量，自动为 GPU 编译。
- **结构化 dtype** – 构建复合类型（类似于 NumPy 结构化数组），具有命名字段。
- **NPY 文件 I/O** – 从/向 NumPy `.npy` 文件加载和保存张量（保留形状，支持 `f32`/`i32`）。
- **便捷的 `tensor!` 宏** – 从嵌套字面量创建张量，可选的 dtype 和设备说明符。
- **Python 绑定** – 通过 PyO3 从 Python 使用 ndrs，完全支持自定义 dtype 和操作覆盖。
- **使用自定义内核覆盖运算符** – 用你自己的 CPU/GPU 内核替换内置实现（例如加法），以获得最佳性能。

---

## 🚀 快速上手

在 `Cargo.toml` 中添加：

```toml
[dependencies]
ndrs = "0.4"
```

### 基本 CPU 用法（`tensor!` 宏）

```rust
use ndrs::{Tensor, s, tensor};

fn main() -> Result<(), String> {
    let a = tensor!([[1, -2], [3, 4]]);      // 自动为 i32，CPU
    let b = tensor!([[5, 6], [7, 8]]);
    let a_view = a.into_rc().as_view();
    let b_view = b.into_rc().as_view();
    let c_view = a_view + b_view;
    assert_eq!(c_view.shape(), &[2, 2]);
    let result = c_view.to_vec::<i32>()?;
    assert_eq!(result, vec![6, 4, 10, 12]);
    Ok(())
}
```

### 指定 dtype/device 创建张量

```rust
let t = tensor!([[1, 2], [3, 4]]);               // CPU, i32
let t = tensor!([[1, 2], [3, 4]]; f32);         // CPU, f32
let t = tensor!([[1, 2], [3, 4]]; "cpu");       // CPU，自动推断 dtype
let t = tensor!([[1.0, 2.0], [3.0, 4.0]]; "cuda:0");   // GPU 0, f32
let t = tensor!([[1, 2], [3, 4]]; i32; "cuda:1");       // GPU 1, i32
```

### 使用 CUDA 流的 GPU 用法

```rust
use ndrs::{tensor, cuda};

fn gpu_stream_example() -> Result<(), String> {
    if !cuda::is_available() { return Ok(()); }
    cuda::set_device(0)?;
    let stream1 = cuda::Stream::new(Some(0))?;
    let stream2 = cuda::Stream::new(Some(0))?;

    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");
    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();
    let mut out_gpu = a_view.create_output()?;

    let start = stream1.record()?;
    a_view.add(&b_view, &mut out_gpu)?;
    let end = stream1.record()?;
    stream1.synchronize()?;
    println!("GPU 加法耗时 {:?}", end.elapsed_since(&start)?);

    let event = stream1.record()?;
    stream2.wait_event(&event)?;
    let mut out2_gpu = out_gpu.create_output()?;
    out_gpu.add(&out_gpu, &mut out2_gpu)?;
    stream2.synchronize()?;
    let result = out2_gpu.to_cpu()?.to_vec::<f32>()?;
    assert_eq!(result, vec![12.0, 16.0, 20.0, 24.0]);
    Ok(())
}
```

### NPY 文件 I/O（NumPy 兼容）

```rust
use ndrs::{tensor, load_npy, save_npy};

fn npy_example() -> Result<(), String> {
    let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
    save_npy(&t, "example.npy")?;
    let loaded = load_npy("example.npy")?;
    assert_eq!(loaded.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
```

### 自定义原始数据类型

```rust
use ndrs::{register_dtype, register_add_op, TypeInfo, DType, Device};
use std::sync::Arc;

const DTYPE_MY_TYPE: DType = 1000;

#[repr(C)]
#[derive(Clone, Copy)]
struct MyType { a: i32, b: i32 }

fn mytype_add_op(/* ... */) -> Result<(), String> { /* ... */ }

register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
register_add_op(DTYPE_MY_TYPE, Arc::new(mytype_add_op));
```

### 自定义逐元素内核（GPU）– Rust

定义适用于任意秩和 dtype 张量的逐元素操作，自动为 GPU 编译。

```rust
use ndrs::{tensor, ArcTensorView, cuda, Device, Tensor};
use ndrs::backend::cuda::ElementwiseKernel;
use ndrs::view::TensorViewOps;

let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");
let mut out = Tensor::new_contiguous(vec![2,2], a.dtype(), Device::Cuda(0))?.into_arc();

let a_view = a.into_arc().as_view();
let b_view = b.into_arc().as_view();
let mut out_view = out.as_view();

// 简单加法
ArcTensorView::elementwise_kernel(&mut out_view, "out = a + b", vec![&a_view, &b_view])?;

// 带局部变量的多语句
ArcTensorView::elementwise_kernel(
    &mut out_view,
    "float tmp = a * 2.0; out = tmp + b",
    vec![&a_view, &b_view],
)?;
```

### Python 自定义逐元素内核

Python 绑定提供了类似 CuPy 的 `ElementwiseKernel`：

```python
import ndrs as nd
import numpy as np

kernel = nd.cuda.ElementwiseKernel(
    "X x, Y y",          # 输入参数：类型占位符 X，变量名 x
    "Z z",               # 输出参数：类型占位符 Z，变量名 z
    "z = (x - y) * (x - y)",  # 表达式
    "squared_diff"       # 内核名称（可选）
)

a = nd.Tensor([1.0, 2.0, 3.0], dtype=nd.float32, device="cuda:0")
b = nd.Tensor([4.0, 5.0, 6.0], dtype=nd.float32, device="cuda:0")
c = kernel(a, b)
np.testing.assert_allclose(c.contiguous().numpy(), [9.0, 9.0, 9.0])
```

多语句示例：

```python
kernel = nd.cuda.ElementwiseKernel(
    "X x, Y y", "Z z",
    "X a = x + 1; Y b = y * 2; z = a + b",
    "multi_stmt"
)
```

内核会自动处理广播、步长以及每个张量的偏移（例如来自切片）。类型占位符（`X`、`Y`、`Z`）在调用时映射到实际的 dtype。

### 底层 RawKernel（动态 CUDA 内核）

```rust
use ndrs::cuda::{RawKernel, get_stream, set_device};
use cudarc::driver::LaunchConfig;
use ndrs::tensor::Tensor;

fn raw_kernel_example() -> anyhow::Result<()> {
    set_device(0)?;
    let stream = get_stream()?;
    let ctx = stream.inner().context().clone();

    let kernel_src = r#"
        extern "C" __global__ void times_two(float* out, const float* in, const int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = 2.0f * in[i];
        }
    "#;
    let kernel = RawKernel::from_source(kernel_src, "times_two", &ctx)?;

    let in_tensor = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let in_gpu = in_tensor.into_arc().as_view().to_gpu(0)?;
    let mut out_gpu = in_gpu.create_output()?;

    let n = in_gpu.size();
    let block = 256;
    let grid = (n + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

    let mut builder = kernel.launch_builder(stream.inner());
    builder.arg(&mut out_gpu.handle().0.lock().borrow_mut().as_gpu_slice_mut().unwrap());
    builder.arg(&in_gpu.handle().0.lock().borrow().as_gpu_slice().unwrap());
    builder.arg(&n);
    unsafe { builder.launch(cfg)?; }

    let result = out_gpu.to_cpu()?.to_vec::<f32>()?;
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
    Ok(())
}
```

---

## 🧠 核心概念

### `Tensor`
原始数据容器。它拥有一个连续的字节缓冲区（CPU 或 GPU）并存储形状、步长、数据类型和设备信息。**不直接实现操作** – 请使用 `TensorView` 进行运算。

构造函数：
- `Tensor::new_cpu_from_slice<T>(&[T], shape)` – 从切片创建。
- `Tensor::new_from_bytes(bytes, shape, dtype, device)` – 从原始字节创建（CPU 或 GPU）。
- `Tensor::new_contiguous(shape, dtype)` – 零初始化的 CPU 张量。
- `Tensor::from_string_literal(s)` – 从字面字符串解析（由 `tensor!` 内部使用）。

### `TensorView`
带有可选偏移、形状和步长的 `Tensor` 视图。所有数学运算（加法、切片、广播、设备传输）都在视图上定义。

提供两种具体的视图类型：

- **`RcTensorView`** – 线程局部变体，使用 `Rc<RefCell<Tensor>>`。  
  适用于单线程代码，快速轻量。
- **`ArcTensorView`** – 线程安全变体，使用 `Arc<ReentrantMutex<RefCell<Tensor>>>`。  
  用于多线程环境和 Python 绑定。

### 切片宏 `s!`
```rust
let sub = view.slice(&s![1..4, 2..6])?;
let row = view.slice(&s![2, ..])?;
let every_other = view.slice(&s![0..8:2, ..])?;
```

### 广播
```rust
use ndrs::broadcast_shapes;
let target = broadcast_shapes(a.shape(), b.shape()).unwrap();
let a_bcast = a_view.broadcast_to(&target)?;
```

### 设备管理
- `Device::Cpu` / `Device::Cuda(id)`
- `cuda::set_device(id)`、`cuda::get_device_count()`、`cuda::get_stream()`、`cuda::set_stream()`

### GPU 流和事件
```rust
let stream = cuda::Stream::new(Some(0))?;
let event = stream.record()?;
stream2.wait_event(&event)?;
```

---

## 📦 Cargo 特性

- **default** – 仅 CPU。
- **cuda** – 启用 GPU 支持（需要 CUDA 工具包和 `cudarc`）。  
  启用 `cuda::*` 函数、`ArcTensorView` GPU 传输、GPU 加速加法以及 `ElementwiseKernel` / `RawKernel` 设施。

---

## 🧪 测试

```bash
cargo test               # 仅 CPU 测试
cargo test -- --ignored  # 包括 GPU 测试（如果 CUDA 可用）
```

---

## 🐍 Python 绑定

`ndrs-python` crate 使用 PyO3 提供 Python 绑定。

### 安装

```bash
cd python
maturin develop          # 仅 CPU
maturin develop --features cuda   # 启用 CUDA 支持
```

### 基本用法

```python
import ndrs as nd
import numpy as np

t = nd.Tensor([[1, 2], [3, 4]], dtype=nd.float32)
t2 = t.to("cuda:0")
t3 = t2 + t2
print(t3.numpy())   # [[2. 4.], [6. 8.]]
```

### Python 中的 CUDA 流和事件

```python
import ndrs as nd

nd.cuda.set_device("cuda:0")
stream1 = nd.cuda.Stream(device_id=0)
stream2 = nd.cuda.Stream(device_id=0)

a = nd.Tensor([1.0, 2.0, 3.0], dtype=nd.float32, device="cuda:0")
b = nd.Tensor([4.0, 5.0, 6.0], dtype=nd.float32, device="cuda:0")

nd.cuda.set_stream(stream1)
c = a + b
event = stream1.record_event()

nd.cuda.set_stream(stream2)
stream2.wait_event(event)
d = c * 2.0
stream2.synchronize()

print(d.numpy())  # [10. 14. 18.]
```

### 自定义 dtype 和操作

**结构化 dtype**：
```python
complex_dtype = nd.dtype.from_fields([('re', nd.float32), ('im', nd.float32)])
data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=complex_dtype.to_numpy_dtype())
t = nd.Tensor(data, dtype=complex_dtype)
```

**覆盖加法**：
```python
def my_add(a_ptr, b_ptr, out_ptr, n, device_code, stream):
    import ctypes, numpy as np
    a = np.ctypeslib.as_array(ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    b = np.ctypeslib.as_array(ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    out = np.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    np.add(a, b, out=out)
    return 0

nd.register_binary_op(nd.float32, nd.BINARY_OP_ADD, "cpu", my_add)
```

### Python API 参考

| 函数 / 类 | 描述 |
|----------|------|
| `nd.Tensor(data, dtype=None, device=None)` | 从列表或 NumPy 数组创建张量。 |
| `nd.Tensor.from_numpy(array, device=None)` | 从 NumPy 数组创建张量的替代构造函数。 |
| `tensor.shape` / `tensor.dtype` / `tensor.device` | 基本属性。 |
| `tensor.numpy()` | 将数据复制到 NumPy 数组。 |
| `tensor.to(device)` | 将张量移动到另一个设备。 |
| `nd.dtype.from_fields(fields)` | 创建结构化 dtype。 |
| `nd.register_dtype(name, itemsize)` | 注册普通 dtype，返回 id。 |
| `nd.register_binary_op(dtype, kind, device, callback)` | 覆盖二元运算（例如 `nd.BINARY_OP_ADD`）。 |
| `nd.cuda.get_device()`、`set_device(device_str)` | 获取/设置当前 CUDA 设备。 |
| `nd.cuda.Stream(device_id)` | 创建 CUDA 流。 |
| `nd.cuda.Event(device_id)` | 创建 CUDA 事件。 |

---

## ⚙️ 性能考虑

- **视图**成本低廉：仅复制形状、步长、偏移和一个句柄。
- **跨步复制**针对 CPU 和 GPU 进行了优化；非连续复制使用迭代内核，但仍然高效。
- **GPU 加法**使用高度优化的步长感知 CUDA 内核。
- **ElementwiseKernel** 为给定的表达式、形状和 dtype 编译专用内核；后续相同签名的调用重用已编译的内核。
- **自动事件跟踪**默认启用，以确保安全的多流同步；如果你手动管理依赖关系，可以禁用它以获得最大吞吐量。

---

## 📄 许可证

本项目采用 **MIT 许可证**。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎贡献！请在 [GitHub](https://github.com/yourusername/ndrs) 上提出问题或拉取请求。对于重大更改，请先讨论。

---

## 🙏 致谢

- 灵感来自 NumPy、PyTorch 和 `ndarray` crate。
- 使用 [cudarc](https://crates.io/crates/cudarc) 进行 CUDA 绑定，并使用 [bytemuck](https://crates.io/crates/bytemuck) 进行安全的字节转换。