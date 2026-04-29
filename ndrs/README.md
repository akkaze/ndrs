# ndrs

[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README_zh.md)

**ndrs** is a NumPy‑like tensor library for Rust, providing multi‑dimensional array (tensor) operations with **optional GPU acceleration** via CUDA. It emphasizes **zero‑copy views**, efficient strided operations, and a flexible ownership model.

---

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Basic CPU usage](#basic-cpu-usage-with-the-tensor-macro)
  - [Creating tensors with dtype/device](#creating-tensors-with-explicit-dtype-and-device)
  - [GPU usage with CUDA streams](#gpu-usage-with-cuda-streams)
  - [NPY file I/O](#npy-file-io-numpy-compatibility)
  - [Custom primitive data types](#custom-primitive-data-types)
  - [Custom elementwise kernels (GPU)](#custom-elementwise-kernels-gpu)
  - [Low‑level RawKernel](#low‑level-rawkernel-dynamic-cuda-kernels)
- [Core Concepts](#-core-concepts)
- [Cargo Features](#-cargo-features)
- [Testing](#-testing)
- [Python Bindings](#-python-bindings)
  - [Installation](#installation)
  - [Basic usage](#basic-usage)
  - [CUDA streams and events from Python](#cuda-streams-and-events-from-python)
  - [Custom dtypes and operations](#custom-dtypes-and-operations)
  - [Python API reference](#python-api-reference)
- [Performance Considerations](#️-performance-considerations)
- [License](#-license)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **N‑dimensional tensors** – shape, strides, and byte‑level data storage.
- **View‑based operations** – slicing, broadcasting, transposing, and reshaping without copying data.
- **Efficient strided copy** – fast data movement between non‑contiguous layouts.
- **Thread‑local and thread‑safe variants**  
  - `Rc<RefCell<Tensor>>` for single‑threaded speed  
  - `Arc<ReentrantMutex<RefCell<Tensor>>>` for multi‑threading and Python bindings
- **GPU acceleration** – transparent CPU ↔ GPU transfer, CUDA kernels for element‑wise operations.
- **CUDA stream support** – asynchronous execution, events for timing and cross‑stream synchronization.
- **Operator overloading** – `+` and `+=` for tensors with broadcastable shapes.
- **Python‑like slicing** – intuitive `s!` macro: `s![1..4:2, ..]`.
- **Broadcasting** – automatic shape expansion.
- **Dynamic CUDA kernels** – compile and launch kernels from PTX or CUDA C++ source at runtime with `RawKernel`.
- **Custom elementwise kernels** – define your own per‑element operations (e.g., `out = a + b * c`) that work on tensors of any rank and dtype, automatically compiled for GPU.
- **Structured dtypes** – build compound types (similar to NumPy structured arrays) with named fields.
- **NPY file I/O** – load and save tensors from/to NumPy `.npy` files (preserving shape, supports `f32`/`i32`).
- **Convenient `tensor!` macro** – create tensors from nested literals with optional dtype and device specifiers.
- **Python bindings** – use ndrs from Python via PyO3, with full support for custom dtypes and operation overriding.
- **Override operators with custom kernels** – replace built‑in implementations (e.g., addition) with your own CPU/GPU kernels for maximum performance.

---

## 🚀 Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ndrs = "0.4"
```

### Basic CPU usage with the `tensor!` macro

```rust
use ndrs::{Tensor, s, tensor};

fn main() -> Result<(), String> {
    let a = tensor!([[1, -2], [3, 4]]);      // automatically i32 on CPU
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

### Creating tensors with explicit dtype and device

```rust
let t = tensor!([[1, 2], [3, 4]]);               // CPU, i32
let t = tensor!([[1, 2], [3, 4]]; f32);         // CPU, f32
let t = tensor!([[1, 2], [3, 4]]; "cpu");       // CPU, auto‑dtype
let t = tensor!([[1.0, 2.0], [3.0, 4.0]]; "cuda:0");   // GPU 0, f32
let t = tensor!([[1, 2], [3, 4]]; i32; "cuda:1");       // GPU 1, i32
```

### GPU usage with CUDA streams

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
    println!("GPU addition took {:?}", end.elapsed_since(&start)?);

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

### NPY file I/O (NumPy compatibility)

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

### Custom primitive data types

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

### Custom elementwise kernels (GPU) – Rust

Define per‑element operations that work on tensors of any rank and dtype, automatically compiled for GPU.

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

// Simple addition
ArcTensorView::elementwise_kernel(&mut out_view, "out = a + b", vec![&a_view, &b_view])?;

// Multi‑statement with local variables
ArcTensorView::elementwise_kernel(
    &mut out_view,
    "float tmp = a * 2.0; out = tmp + b",
    vec![&a_view, &b_view],
)?;
```

### Python custom elementwise kernels

The Python binding provides a CuPy‑like `ElementwiseKernel`:

```python
import ndrs as nd
import numpy as np

kernel = nd.cuda.ElementwiseKernel(
    "X x, Y y",          # input parameters: type placeholder X, variable name x
    "Z z",               # output parameter: type placeholder Z, variable name z
    "z = (x - y) * (x - y)",  # expression
    "squared_diff"       # kernel name (optional)
)

a = nd.Tensor([1.0, 2.0, 3.0], dtype=nd.float32, device="cuda:0")
b = nd.Tensor([4.0, 5.0, 6.0], dtype=nd.float32, device="cuda:0")
c = kernel(a, b)
np.testing.assert_allclose(c.contiguous().numpy(), [9.0, 9.0, 9.0])
```

Multi‑statement example:

```python
kernel = nd.cuda.ElementwiseKernel(
    "X x, Y y", "Z z",
    "X a = x + 1; Y b = y * 2; z = a + b",
    "multi_stmt"
)
```

The kernel automatically handles broadcasting, strides, and per‑tensor offsets (e.g., from slicing). Type placeholders (`X`, `Y`, `Z`) are mapped to actual dtypes at call time.

### Low‑level RawKernel (dynamic CUDA kernels)

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

## 🧠 Core Concepts

### `Tensor`
The raw data container. It owns a contiguous byte buffer (either on CPU or GPU) and stores shape, strides, data type, and device information. **It does not implement operations directly** – use `TensorView` for that.

Constructors:
- `Tensor::new_cpu_from_slice<T>(&[T], shape)` – from slice.
- `Tensor::new_from_bytes(bytes, shape, dtype, device)` – from raw bytes (CPU or GPU).
- `Tensor::new_contiguous(shape, dtype)` – zero‑initialized CPU tensor.
- `Tensor::from_string_literal(s)` – parse from a literal string (used by `tensor!`).

### `TensorView`
A view into a `Tensor` with an optional offset, shape, and strides. All mathematical operations (addition, slicing, broadcasting, device transfer) are defined on views.

Two concrete view types are provided:

- **`RcTensorView`** – thread‑local variant using `Rc<RefCell<Tensor>>`.  
  Fast and lightweight for single‑threaded code.
- **`ArcTensorView`** – thread‑safe variant using `Arc<ReentrantMutex<RefCell<Tensor>>>`.  
  Required for multi‑threaded environments and Python bindings.

### Slice macro `s!`
```rust
let sub = view.slice(&s![1..4, 2..6])?;
let row = view.slice(&s![2, ..])?;
let every_other = view.slice(&s![0..8:2, ..])?;
```

### Broadcasting
```rust
use ndrs::broadcast_shapes;
let target = broadcast_shapes(a.shape(), b.shape()).unwrap();
let a_bcast = a_view.broadcast_to(&target)?;
```

### Device management
- `Device::Cpu` / `Device::Cuda(id)`
- `cuda::set_device(id)`, `cuda::get_device_count()`, `cuda::get_stream()`, `cuda::set_stream()`

### GPU streams and events
```rust
let stream = cuda::Stream::new(Some(0))?;
let event = stream.record()?;
stream2.wait_event(&event)?;
```

---

## 📦 Cargo Features

- **default** – CPU only.
- **cuda** – enables GPU support (requires CUDA toolkit and `cudarc`).  
  Enables `cuda::*` functions, `ArcTensorView` GPU transfers, GPU‑accelerated addition, and the `ElementwiseKernel` / `RawKernel` facilities.

---

## 🧪 Testing

```bash
cargo test               # CPU tests only
cargo test -- --ignored  # includes GPU tests (if CUDA available)
```

---

## 🐍 Python Bindings

The `ndrs-python` crate provides Python bindings using PyO3.

### Installation

```bash
cd python
maturin develop          # CPU only
maturin develop --features cuda   # with CUDA support
```

### Basic usage

```python
import ndrs as nd
import numpy as np

t = nd.Tensor([[1, 2], [3, 4]], dtype=nd.float32)
t2 = t.to("cuda:0")
t3 = t2 + t2
print(t3.numpy())   # [[2. 4.], [6. 8.]]
```

### CUDA streams and events from Python

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

### Custom dtypes and operations

**Structured dtypes**:
```python
complex_dtype = nd.dtype.from_fields([('re', nd.float32), ('im', nd.float32)])
data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=complex_dtype.to_numpy_dtype())
t = nd.Tensor(data, dtype=complex_dtype)
```

**Override addition**:
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

### Python API reference

| Function / Class | Description |
|----------------|-------------|
| `nd.Tensor(data, dtype=None, device=None)` | Create tensor from list or NumPy array. |
| `nd.Tensor.from_numpy(array, device=None)` | Alternative constructor from NumPy array. |
| `tensor.shape` / `tensor.dtype` / `tensor.device` | Basic properties. |
| `tensor.numpy()` | Copy data to NumPy array. |
| `tensor.to(device)` | Move tensor to another device. |
| `nd.dtype.from_fields(fields)` | Create structured dtype. |
| `nd.register_dtype(name, itemsize)` | Register plain dtype, returns id. |
| `nd.register_binary_op(dtype, kind, device, callback)` | Override binary op (e.g., `nd.BINARY_OP_ADD`). |
| `nd.cuda.get_device()`, `set_device(device_str)` | Get/set current CUDA device. |
| `nd.cuda.Stream(device_id)` | Create a CUDA stream. |
| `nd.cuda.Event(device_id)` | Create a CUDA event. |

---

## ⚙️ Performance Considerations

- **Views** are cheap: they copy only shape, strides, offset, and a handle.
- **Strided copy** is optimized for CPU and GPU; non‑contiguous copies use an iterative kernel but are still efficient.
- **GPU addition** uses a highly optimized stride‑aware CUDA kernel.
- **ElementwiseKernel** compiles a dedicated kernel for the given expression, shape, and dtypes; subsequent calls with the same signature reuse the compiled kernel.
- **Automatic event tracking** is enabled by default to ensure safe multi‑stream synchronization; you can disable it for maximum throughput if you manage dependencies manually.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/yourusername/ndrs). For major changes, please discuss first.

---

## 🙏 Acknowledgments

- Inspired by NumPy, PyTorch, and the `ndarray` crate.
- Uses [cudarc](https://crates.io/crates/cudarc) for CUDA bindings and [bytemuck](https://crates.io/crates/bytemuck) for safe byte casts.