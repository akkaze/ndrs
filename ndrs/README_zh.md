# ndrs

[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README_zh.md)

**ndrs** 是一个类 NumPy 的 Rust 张量库，提供多维数组（张量）操作，并支持通过 CUDA 实现可选的 GPU 加速。它强调**零拷贝视图**、高效的跨步操作以及灵活的所有权模型。

---

## ✨ 特性

- **N 维张量** – 形状、步长和字节级数据存储。
- **基于视图的操作** – 切片、广播、转置和重塑，无需复制数据。
- **高效的跨步复制** – 非连续布局之间的快速数据移动。
- **线程局部和线程安全变体**  
  - `Rc<RefCell<Tensor>>` – 单线程高性能  
  - `Arc<ReentrantMutex<RefCell<Tensor>>>` – 多线程及 Python 绑定
- **GPU 加速** – 透明的 CPU ↔ GPU 传输，CUDA 核函数用于逐元素加法。
- **CUDA 流支持** – 异步执行，事件用于计时和跨流同步（`wait_event`）。
- **运算符重载** – 支持 `+` 和 `+=` 运算，形状可广播。
- **类 Python 的切片宏** – `s!` 宏：`s![1..4:2, ..]`。
- **广播** – 自动形状扩展。
- **自定义数据类型** – 注册自己的原始类型或结构化类型，并自定义加法运算。
- **结构化 dtype** – 构建复合类型（类似 NumPy 结构化数组），支持命名字段。
- **NPY 文件 I/O** – 从/向 NumPy `.npy` 文件加载和保存张量（保持形状，支持 `f32`/`i32`）。
- **便捷的 `tensor!` 宏** – 从嵌套字面量创建张量，可指定 dtype 和设备。
- **Python 绑定** – 通过 PyO3 使用 ndrs，完全支持自定义 dtype 和运算覆盖。
- **用自定义 kernel 覆盖运算符** – 用自己的 CPU/GPU kernel（如加法）替换内置实现，以获得最佳性能。

---

## 🚀 快速开始

在 `Cargo.toml` 中添加：

```toml
[dependencies]
ndrs = "0.4"
```

### 基本 CPU 用法与 `tensor!` 宏

```rust
use ndrs::{Tensor, s, tensor};

fn main() -> Result<(), String> {
    // 使用便捷的 `tensor!` 宏创建张量（支持负数、浮点数）
    let a = tensor!([[1, -2], [3, 4]]);      // 自动 i32 类型，CPU
    let b = tensor!([[5, 6], [7, 8]]);       // i32，形状 [2,2]

    // 包装成线程局部共享视图（Rc<RefCell<Tensor>>）
    let a_view = a.into_rc().as_view();
    let b_view = b.into_rc().as_view();

    // 使用 '+' 运算符（形状不同时会自动广播）
    let c_view = a_view + b_view;
    assert_eq!(c_view.shape(), &[2, 2]);

    // 转换为 `Vec<i32>`
    let result = c_view.to_vec::<i32>()?;
    assert_eq!(result, vec![6, 4, 10, 12]);

    // 就地加法 `+=`
    let mut a_mut = a_view.clone();
    a_mut += b_view;
    assert_eq!(a_mut.to_vec::<i32>()?, vec![6, 4, 10, 12]);

    // 也可以直接调用底层的 `add` 方法（`+` 运算符使用的）
    let mut out = a_view.create_output()?;   // 全零
    a_view.add(&b_view, &mut out)?;          // 计算 out = a + b
    assert_eq!(out.to_vec::<i32>()?, vec![6, 4, 10, 12]);

    Ok(())
}
```

### 显式指定 dtype 和设备创建张量

`tensor!` 宏支持可选的 `; dtype` 和 `; "device"` 说明符：

```rust
let t = tensor!([[1, 2], [3, 4]]);               // CPU, i32
let t = tensor!([[1, 2], [3, 4]]; f32);         // CPU, f32
let t = tensor!([[1, 2], [3, 4]]; "cpu");       // CPU, 自动推断类型
let t = tensor!([[1.0, 2.0], [3.0, 4.0]]; "cuda:0");   // GPU 0, f32
let t = tensor!([[1, 2], [3, 4]]; i32; "cuda:1");       // GPU 1, i32
```

### GPU 使用与 CUDA 流

```rust
use ndrs::{tensor, cuda};

fn gpu_stream_example() -> Result<(), String> {
    // 确保 CUDA 可用
    if !cuda::is_available() { return Ok(()); }
    cuda::set_device(0)?;

    // 创建两个流
    let stream1 = cuda::Stream::new(Some(0))?;
    let stream2 = cuda::Stream::new(Some(0))?;

    // 直接在 GPU 上创建张量（可选）
    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");

    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();
    let mut out_gpu = a_view.create_output()?;   // 同一设备上的全零

    // 记录事件以测量内核时间
    let start = stream1.record()?;
    a_view.add(&b_view, &mut out_gpu)?;
    let end = stream1.record()?;
    stream1.synchronize()?;
    let elapsed = end.elapsed_since(&start)?;
    println!("GPU 加法耗时: {:?}", elapsed);

    // 流2等待流1的事件再开始
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

可以注册自己的原始（非结构化）类型并定义加法：

```rust
use ndrs::{register_dtype, register_add_op, TypeInfo, DType, Device};
use std::sync::Arc;

const DTYPE_MY_TYPE: DType = 1000;

#[repr(C)]
#[derive(Clone, Copy)]
struct MyType { a: i32, b: i32 }

fn mytype_add_op(
    a: *const u8, a_strides: *const usize,
    b: *const u8, b_strides: *const usize,
    c: *mut u8, c_strides: *const usize,
    shape: *const usize, ndim: usize, n: usize,
    dev: Device, stream: Option<*mut std::ffi::c_void>
) -> Result<(), String> {
    // 使用步长和设备区分实现
    // ...
    Ok(())
}

// 注册
register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
register_add_op(DTYPE_MY_TYPE, Arc::new(mytype_add_op));
```

### 自定义结构化 dtype（Python）

在 Python 绑定中，可以定义类似 NumPy 结构化数组的**结构化 dtype**：

```python
import ndrs as nd
import numpy as np

# 定义由两个 float32 字段组成的复数 dtype
complex_dtype = nd.dtype.from_fields([
    ('re', nd.float32),
    ('im', nd.float32)
])

# 从 NumPy 结构化数组创建张量
data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=complex_dtype.to_numpy_dtype())
t = nd.Tensor(data, dtype=complex_dtype)

# 转回 NumPy 后访问字段
arr = t.numpy()
print(arr['re'])  # [1.0, 3.0]
print(arr['im'])  # [2.0, 4.0]
```

---

## 🧠 核心概念

### `Tensor`
原始数据容器。拥有一个连续的字节缓冲区（CPU 或 GPU），存储形状、步长、数据类型和设备信息。**本身不实现运算** – 请使用 `TensorView` 进行运算。

构造函数：
- `Tensor::new_cpu_from_slice<T>(&[T], shape)` – 从切片创建。
- `Tensor::new_from_bytes(bytes, shape, dtype, device)` – 从原始字节创建（CPU 或 GPU）。
- `Tensor::new_contiguous(shape, dtype)` – 零初始化的 CPU 张量。
- `Tensor::from_string_literal(s)` – 从字符串字面量解析（被 `tensor!` 宏使用）。

### `TensorView`
指向 `Tensor` 的视图，包含可选的偏移、形状和步长。所有数学运算（加法、切片、广播、设备传输）都定义在视图上。

提供两种具体的视图类型：

- **`RcTensorView`** – 线程局部变体，基于 `Rc<RefCell<Tensor>>`。  
  单线程代码快速轻量，所有操作非阻塞且成本低。
- **`ArcTensorView`** – 线程安全变体，基于 `Arc<ReentrantMutex<RefCell<Tensor>>>`。  
  多线程环境和 Python 绑定所需。锁自动且可重入。

### 切片宏 `s!`
为 `.slice()` 方法创建切片描述符。支持范围、步长、单索引和 `..`（全选）。

```rust
let sub = view.slice(&s![1..4, 2..6])?;       // 行 1..4，列 2..6
let row = view.slice(&s![2, ..])?;            // 单行（降维）
let col = view.slice(&s![.., 3])?;            // 单列
let every_other = view.slice(&s![0..8:2, ..])?; // 每隔一行
```

### 广播
使用 `broadcast_shapes` 计算两个张量的目标形状，然后用 `broadcast_to` 扩展视图。

```rust
use ndrs::broadcast_shapes;

let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
let b = Tensor::new_cpu_from_f32(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
let target = broadcast_shapes(a.shape(), b.shape()).unwrap(); // [3, 4]
let a_bcast = a_view.broadcast_to(&target)?;
let b_bcast = b_view.broadcast_to(&target)?;
```

### 设备管理

- `Device::Cpu` – 主机内存。
- `Device::Cuda(id)` – 给定索引的 CUDA 设备。
- `cuda::set_device(id)` – 设置创建上下文时的默认设备。
- `cuda::get_device_count()` – 返回 CUDA 设备数量。
- `cuda::get_stream()` / `cuda::set_stream()` – 线程局部当前 CUDA 流。

### GPU 流和事件

- **流** 支持异步命令提交。使用 `cuda::Stream::new()` 创建自定义非默认流。
- **事件** 记录流中的时间点，可用于计时（`elapsed_since`）或跨流依赖（`wait_event`）。

```rust
let stream = cuda::Stream::new(Some(0))?;
// ... 启动内核，复制数据 ...
let event = stream.record()?;
stream2.wait_event(&event)?; // 等待另一个流
```

---

## 📦 Cargo Features

- **default** – 仅 CPU。
- **cuda** – 启用 GPU 支持（需要 CUDA 工具包和 `cudarc`）。  
  启用 `cuda::*` 函数、`ArcTensorView` GPU 传输和 GPU 加速加法。

---

## 🧪 测试

运行所有测试（仅 CPU；若无设备，GPU 测试默认被忽略）：

```bash
cargo test
```

运行 GPU 测试（需要 CUDA 设备）：

```bash
cargo test -- --ignored
```

---

## 🐍 Python 绑定

`ndrs-python` crate 使用 PyO3 提供 Python 绑定。从源码安装：

```bash
cd python
maturin develop
```

然后在 Python 中：

```python
import ndrs as nd

# 从嵌套列表创建张量（自动检测 dtype）
t = nd.Tensor([[1, 2], [3, 4]], dtype=nd.float32)

# 移到 GPU 并相加
t2 = t.to("cuda:0")
t3 = t2 + t2

# 转换回 NumPy（需要安装 `numpy`）
print(t3.numpy())   # [[2. 4.]
                    #  [6. 8.]]
```

### 从 Python 自定义运算（覆盖内置 kernel）

你可能希望用自己的高性能 kernel 替换 ndrs 对某个二元运算（例如加法）的默认实现 —— 无论是针对内置 dtype（如 `float32`）还是自定义 dtype。

`register_binary_op` 函数允许你提供一个 Python 回调函数，该回调会在指定的 dtype、运算和设备上被调用。回调接收输入/输出缓冲区的原始指针、元素数量、设备代码和可选的流指针。你可以使用 `ctypes` + `numpy` 访问数据并执行计算。

对于**性能敏感**的自定义 kernel，你可以编写 C/CUDA 函数，编译成共享库，然后在 Python 中通过 `ctypes` 调用。这样能完全控制 kernel 而无需牺牲性能。

**示例：用更快的（向量化）实现替换 CPU 上的 `float32` 加法**

```python
import ctypes
import numpy as np
import ndrs as nd

def fast_float32_add(a_ptr, b_ptr, out_ptr, n, device_code, stream):
    # 假设数据是连续的（可以在内核中调用 .contiguous() 或要求用户先做）
    # 使用 numpy 进行 SIMD 优化加法（或调用自己的 C 库）
    a = np.ctypeslib.as_array(ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    b = np.ctypeslib.as_array(ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    out = np.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    np.add(a, b, out=out)          # NumPy 的向量化加法
    # 可选：再乘以 2 等
    return 0  # 成功

# 覆盖 float32 在 CPU 上的默认加法
nd.register_binary_op(nd.float32, nd.BINARY_OP_ADD, "cpu", fast_float32_add)
```

**对于 CUDA**，你可以编写 `.ptx` 或 `.cubin` 内核，通过 `ctypes` 或 `cupy` 加载，并在回调中调用它。

### 从 Rust 覆盖

Rust API 同样允许覆盖运算。当你希望集成直接用 Rust 编写的 kernel（例如使用 `ndarray` 或 `rayon`）或第三方 CUDA kernel 时，这很有用。

```rust
use ndrs::{register_binary_op, BinaryOpKind, Device, BinaryOpFn, DTYPE_FLOAT32};
use std::sync::Arc;

fn my_fast_add_f32_cpu(
    a: *const u8, a_strides: *const usize,
    b: *const u8, b_strides: *const usize,
    c: *mut u8, c_strides: *const usize,
    shape: *const usize, ndim: usize, n: usize,
    dev: Device, stream: Option<*mut std::ffi::c_void>
) -> Result<(), String> {
    // 你的优化实现（例如使用 SIMD 或自定义算法）
    // ...
    Ok(())
}

let op: BinaryOpFn = Arc::new(my_fast_add_f32_cpu);
register_binary_op(DTYPE_FLOAT32, BinaryOpKind::Add, Device::Cpu, op);
```

注册后，该 dtype 和设备上的所有张量加法都会使用你的自定义 kernel。

### 自定义 kernel 的重要说明

- **连续性**：kernel 可能被任意步长调用。为简化实现，你可以在 kernel 内部先调用 `.contiguous()`（或要求用户这样做），但这会引入复制开销。为获得最佳性能，你的 kernel 应该能感知步长（如 ndrs 的默认 CPU/GPU kernel）。
- **线程安全**：回调可能被多个线程调用，必须能安全并发使用。
- **设备特定**：可以为 CPU 和 CUDA 注册不同的 kernel，从而在保留 CPU 备用方案的同时使用专用 GPU kernel。
- **性能提升**：通过覆盖内置运算，你可以集成手工调优的 CPU 向量化（如使用 `avx2` 内部函数）或高度优化的 CUDA kernel（如使用张量核心），无需等待 ndrs 原生支持。

### Python API 参考

| 函数 / 类                            | 描述                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| `nd.Tensor(data, dtype=None, device=None)` | 从 Python 列表或 NumPy 数组创建张量。                         |
| `nd.Tensor.from_numpy(array, device=None)` | 从 NumPy 数组创建张量的替代构造器。                           |
| `tensor.shape`                       | 返回形状（整数列表）。                                       |
| `tensor.dtype`                       | 返回 dtype id（整数）或自定义 dtype 的 `DType` 对象。        |
| `tensor.device`                      | 返回设备字符串（例如 `"cpu"`, `"cuda:0"`）。                 |
| `tensor.numpy()`                     | 返回 NumPy 数组（副本）。                                   |
| `tensor.to(device)`                  | 将张量移到另一设备（返回新张量）。                           |
| `nd.dtype.from_fields(fields)`        | 创建自定义结构化 dtype。`fields` 是 `(name, dtype_id)` 列表。 |
| `nd.register_dtype(name, itemsize)`   | 注册一个普通（非结构化）dtype，返回 dtype id。               |
| `nd.register_binary_op(dtype, kind, device, callback)` | 为特定的 dtype 和设备注册二元运算回调。`kind` 可以是 `nd.BINARY_OP_ADD`、`nd.BINARY_OP_SUB` 等。 |

---

## ⚙️ 性能考量

- **视图**成本低：只复制形状、步长、偏移和底层 `Tensor` 的句柄。
- **跨步复制**针对 CPU 和 GPU 优化；非连续复制使用回退的迭代 kernel，但仍然高效。
- **GPU 加法**使用高度优化的 CUDA kernel，尊重任意步长，达到接近峰值的内存带宽。
- **自动事件跟踪**默认开启，确保多流同步安全；如果手动管理依赖关系，可以禁用事件跟踪以获得最大吞吐量。

---

## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎贡献！请在 [GitHub](https://github.com/yourusername/ndrs) 上提出问题或拉取请求。重大更改请先讨论。

---

## 🙏 致谢

- 灵感来自 NumPy、PyTorch 和 `ndarray` crate。
- 使用 [cudarc](https://crates.io/crates/cudarc) 进行 CUDA 绑定，[bytemuck](https://crates.io/crates/bytemuck) 进行安全的字节转换。
- NPY I/O 由 [npyz](https://crates.io/crates/npyz) 驱动。