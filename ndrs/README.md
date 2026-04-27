# ndrs

[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README_zh.md)

**ndrs** is a NumPy‑like tensor library for Rust, providing multi‑dimensional array (tensor) operations with **optional GPU acceleration** via CUDA. It emphasizes **zero‑copy views**, efficient strided operations, and a flexible ownership model.

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
    // Create tensors using the convenient `tensor!` macro (supports negatives, floats)
    let a = tensor!([[1, -2], [3, 4]]);      // automatically i32 on CPU
    let b = tensor!([[5, 6], [7, 8]]);       // i32, shape [2,2]

    // Wrap into a thread‑local shared view (Rc<RefCell<Tensor>>)
    let a_view = a.into_rc().as_view();
    let b_view = b.into_rc().as_view();

    // Use the '+' operator (broadcasts if shapes differ)
    let c_view = a_view + b_view;
    assert_eq!(c_view.shape(), &[2, 2]);

    // Convert back to a `Vec<i32>`
    let result = c_view.to_vec::<i32>()?;
    assert_eq!(result, vec![6, 4, 10, 12]);

    // In‑place addition with `+=`
    let mut a_mut = a_view.clone();
    a_mut += b_view;
    assert_eq!(a_mut.to_vec::<i32>()?, vec![6, 4, 10, 12]);

    // The underlying `add` method (used by `+`) is also available:
    let mut out = a_view.create_output()?;   // zeros
    a_view.add(&b_view, &mut out)?;          // compute out = a + b
    assert_eq!(out.to_vec::<i32>()?, vec![6, 4, 10, 12]);

    Ok(())
}
```

### Creating tensors with explicit dtype and device

The `tensor!` macro accepts optional `; dtype` and `; "device"` specifiers:

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
    // Ensure CUDA is available
    if !cuda::is_available() { return Ok(()); }
    cuda::set_device(0)?;

    // Create two streams
    let stream1 = cuda::Stream::new(Some(0))?;
    let stream2 = cuda::Stream::new(Some(0))?;

    // Create tensors directly on GPU (optional)
    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");

    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();
    let mut out_gpu = a_view.create_output()?;   // zeros on the same device

    // Record event to measure kernel time
    let start = stream1.record()?;
    a_view.add(&b_view, &mut out_gpu)?;
    let end = stream1.record()?;
    stream1.synchronize()?;
    let elapsed = end.elapsed_since(&start)?;
    println!("GPU addition took {:?}", elapsed);

    // Stream2 waits for stream1's event before starting
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

You can register your own primitive (non‑structured) types and define addition for them:

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
    // Implementation using strides and device discrimination
    // ...
    Ok(())
}

// Register
register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
register_add_op(DTYPE_MY_TYPE, Arc::new(mytype_add_op));
```

### Custom elementwise kernels (GPU)

ndrs provides a high‑level `ElementwiseKernel` that compiles a user‑defined expression (e.g., `"out = a * b + c"`) into a CUDA kernel at runtime. The kernel automatically respects broadcasting and strides.

```rust
use ndrs::{tensor, ArcTensorView, cuda};
use ndrs::builtin_kernels::cuda::ElementwiseKernel;
use ndrs::view::TensorViewOps;
use std::sync::Arc;

fn elementwise_example() -> anyhow::Result<()> {
    cuda::set_device(0)?;
    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");
    let mut out = Tensor::new_contiguous(vec![2,2], DTYPE_FLOAT32, Device::Cuda(0))?.into_arc();

    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();
    let mut out_view = out.as_view();

    // Launch a kernel that computes out = a + b
    ArcTensorView::elementwise_kernel(&mut out_view, "out = a + b", vec![&a_view, &b_view])?;

    // Or use a more complex expression:
    // out = a * b + 2.0 * a
    ArcTensorView::elementwise_kernel(&mut out_view, "out = a * b + 2.0 * a", vec![&a_view, &b_view])?;

    Ok(())
}
```

The `ElementwiseKernel` compiles a dedicated kernel for the exact number of dimensions and data types, achieving near‑optimal performance. It works with both `RcTensorView` and `ArcTensorView`.

好的，我将把 `ElementwiseKernel` 和 `RawKernel` 的说明添加到 README 中。以下是更新后的 README 内容（仅展示新增部分，完整 README 需整合）：

---

### Custom elementwise kernels (GPU)

ndrs provides a high‑level `ElementwiseKernel` that compiles a user‑defined expression (e.g., `"out = a * b + c"`) into a CUDA kernel at runtime. The kernel automatically respects broadcasting and strides.

```rust
use ndrs::{tensor, ArcTensorView, cuda, DTYPE_FLOAT32, Device};
use ndrs::builtin_kernels::cuda::ElementwiseKernel;
use ndrs::view::TensorViewOps;

fn elementwise_example() -> anyhow::Result<()> {
    cuda::set_device(0)?;
    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]; f32; "cuda:0");
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]; f32; "cuda:0");
    let mut out = Tensor::new_contiguous(vec![2,2], DTYPE_FLOAT32, Device::Cuda(0))?.into_arc();

    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();
    let mut out_view = out.as_view();

    // Launch a kernel that computes out = a + b
    ArcTensorView::elementwise_kernel(&mut out_view, "out = a + b", vec![&a_view, &b_view])?;

    // Or use a more complex expression:
    // out = a * b + 2.0 * a
    ArcTensorView::elementwise_kernel(&mut out_view, "out = a * b + 2.0 * a", vec![&a_view, &b_view])?;

    Ok(())
}
```

The `ElementwiseKernel` compiles a dedicated kernel for the exact number of dimensions and data types, achieving near‑optimal performance. It works with both `RcTensorView` and `ArcTensorView`.

### Low‑level RawKernel (dynamic CUDA kernels)

For complete control, use `RawKernel` to load PTX code or compile CUDA C++ source directly inside your Rust program. This is useful when you want to launch custom‑written CUDA kernels without external compilation steps.

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
    let mut out_gpu = in_gpu.create_output()?;  // zeros on GPU

    let n = in_gpu.size();
    let block = 256;
    let grid = (n + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = kernel.launch_builder(stream.inner());
    // `PushKernelArg` is implemented for `&mut CudaSlice<T>`, `&CudaSlice<T>`, etc.
    builder.arg(&mut out_gpu.handle().0.lock().borrow_mut().as_gpu_slice_mut().unwrap());
    builder.arg(&in_gpu.handle().0.lock().borrow().as_gpu_slice().unwrap());
    builder.arg(&n);
    unsafe { builder.launch(cfg)?; }

    let result = out_gpu.to_cpu()?.to_vec::<f32>()?;
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
    Ok(())
}
```

This gives you full flexibility to write and launch any CUDA kernel while still benefiting from ndrs’s tensor views and memory management.

---

### Custom structured dtypes (Python)

In the Python bindings, you can define **structured dtypes** similar to NumPy’s structured arrays:

```python
import ndrs as nd
import numpy as np

# Define a complex dtype composed of two float32 fields
complex_dtype = nd.dtype.from_fields([
    ('re', nd.float32),
    ('im', nd.float32)
])

# Create a tensor from a NumPy structured array
data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=complex_dtype.to_numpy_dtype())
t = nd.Tensor(data, dtype=complex_dtype)

# Access fields after conversion back to NumPy
arr = t.numpy()
print(arr['re'])  # [1.0, 3.0]
print(arr['im'])  # [2.0, 4.0]
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
  Fast and lightweight for single‑threaded code. All operations are non‑blocking and cheap.
- **`ArcTensorView`** – thread‑safe variant using `Arc<ReentrantMutex<RefCell<Tensor>>>`.  
  Required for multi‑threaded environments and Python bindings. Locking is automatic and reentrant.

### Slice macro `s!`
Creates a slice descriptor for the `.slice()` method. Supports ranges, steps, single indices, and `..` (all).

```rust
let sub = view.slice(&s![1..4, 2..6])?;       // rows 1..4, cols 2..6
let row = view.slice(&s![2, ..])?;            // single row (dimension reduced)
let col = view.slice(&s![.., 3])?;            // single column
let every_other = view.slice(&s![0..8:2, ..])?; // every second row
```

### Broadcasting
Use `broadcast_shapes` to compute the target shape for two tensors, then `broadcast_to` to expand a view.

```rust
use ndrs::broadcast_shapes;

let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
let b = Tensor::new_cpu_from_f32(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
let target = broadcast_shapes(a.shape(), b.shape()).unwrap(); // [3, 4]
let a_bcast = a_view.broadcast_to(&target)?;
let b_bcast = b_view.broadcast_to(&target)?;
```

### Device management

- `Device::Cpu` – host memory.
- `Device::Cuda(id)` – CUDA device with given index.
- `cuda::set_device(id)` – sets the default device for context creation.
- `cuda::get_device_count()` – returns number of CUDA‑capable devices.
- `cuda::get_stream()` / `cuda::set_stream()` – thread‑local current CUDA stream.

### GPU streams and events

- **Streams** allow asynchronous command submission. Use `cuda::Stream::new()` to create a custom non‑default stream.
- **Events** record points in a stream, can be used for timing (`elapsed_since`) or cross‑stream dependencies (`wait_event`).

```rust
let stream = cuda::Stream::new(Some(0))?;
// ... launch kernels, copy data ...
let event = stream.record()?;
stream2.wait_event(&event)?; // wait on another stream
```

---

## 📦 Cargo Features

- **default** – CPU only.
- **cuda** – enables GPU support (requires CUDA toolkit and `cudarc`).  
  Enables `cuda::*` functions, `ArcTensorView` GPU transfers, GPU‑accelerated addition, and the `ElementwiseKernel` / `RawKernel` facilities.

---

## 🧪 Testing

Run all tests (CPU only, GPU tests are ignored by default if no device):

```bash
cargo test
```

To run GPU tests (requires CUDA device):

```bash
cargo test -- --ignored
```

---

## 🐍 Python Bindings

The `ndrs-python` crate provides Python bindings using PyO3. Install from source:

```bash
cd python
maturin develop
```

Then in Python:

```python
import ndrs as nd

# Create a tensor from a nested list (auto‑detects dtype)
t = nd.Tensor([[1, 2], [3, 4]], dtype=nd.float32)

# Move to GPU and add
t2 = t.to("cuda:0")
t3 = t2 + t2

# Convert back to NumPy (requires `numpy` installed)
print(t3.numpy())   # [[2. 4.]
                    #  [6. 8.]]
```

### Customizing operations from Python (overriding built‑in kernels)

You may want to replace ndrs’s default implementation of a binary operation (e.g., `Add`) with your own highly optimized kernel – either for a built‑in dtype like `float32` or for a custom dtype.

The `register_binary_op` function allows you to supply a Python callback that will be invoked for the given dtype, operation, and device. The callback receives raw pointers to the input and output buffers, the number of elements, a device code, and an optional stream pointer. You can use `ctypes` + `numpy` to access the data and perform the computation.

For **performance‑critical** custom kernels, you can write a C/CUDA function, compile it into a shared library, then call it from Python via `ctypes`. This gives you full control over the kernel without sacrificing speed.

**Example: Replace the CPU addition for `float32` with a faster (or vectorized) implementation**

```python
import ctypes
import numpy as np
import ndrs as nd

def fast_float32_add(a_ptr, b_ptr, out_ptr, n, device_code, stream):
    # Assume the data is contiguous (ndrs will pass contiguous tensors if you call .contiguous() first)
    # Use numpy for SIMD-optimized addition (or call your own C library)
    a = np.ctypeslib.as_array(ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    b = np.ctypeslib.as_array(ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    out = np.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=(n,))
    np.add(a, b, out=out)          # NumPy’s vectorized add
    # Optionally multiply by 2, etc.
    return 0  # success

# Override the default CPU addition for float32
nd.register_binary_op(nd.float32, nd.BINARY_OP_ADD, "cpu", fast_float32_add)
```

**For CUDA**, you can write a `.ptx` or `.cubin` kernel, load it via `ctypes` or `cupy`, and invoke it inside the callback.

### Overriding from Rust

The Rust API also lets you override operations. This is useful when you want to integrate a kernel written directly in Rust (e.g., using `ndarray` or `rayon`) or a third‑party CUDA kernel.

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
    // Your optimized implementation (e.g., using SIMD or a custom algorithm)
    // ...
    Ok(())
}

let op: BinaryOpFn = Arc::new(my_fast_add_f32_cpu);
register_binary_op(DTYPE_FLOAT32, BinaryOpKind::Add, Device::Cpu, op);
```

After registration, all tensor additions using that dtype and device will route through your custom kernel.

### Important notes for custom kernels

- **Contiguity**: The kernel may be called with arbitrary strides. To simplify your implementation, you can first call `.contiguous()` on the tensors inside the kernel (or require the user to do so) – but this will add a copy overhead. For maximum performance, your kernel should be stride‑aware (like ndrs’s default CPU and GPU kernels).
- **Thread safety**: The callback will be called from possibly multiple threads; it must be safe to use concurrently.
- **Device‑specific**: You can register different kernels for CPU and CUDA, allowing you to use specialized GPU kernels while keeping a CPU fallback.
- **Performance gains**: By overriding built‑in operations, you can integrate hand‑tuned CPU vectorization (e.g., using `avx2` intrinsics) or highly optimized CUDA kernels (e.g., using tensor cores) without waiting for ndrs to natively support them.

### Python API reference

| Function / Class | Description |
|----------------|-------------|
| `nd.Tensor(data, dtype=None, device=None)` | Create a tensor from a Python list or NumPy array. |
| `nd.Tensor.from_numpy(array, device=None)` | Alternative constructor from a NumPy array. |
| `tensor.shape` | Returns shape as list of ints. |
| `tensor.dtype` | Returns dtype id (int) or `DType` object for custom dtypes. |
| `tensor.device` | Returns device string (e.g., `"cpu"`, `"cuda:0"`). |
| `tensor.numpy()` | Returns a NumPy array (copy). |
| `tensor.to(device)` | Moves tensor to another device (returns new tensor). |
| `nd.dtype.from_fields(fields)` | Create a custom structured dtype. `fields` is a list of `(name, dtype_id)`. |
| `nd.register_dtype(name, itemsize)` | Register a plain (non‑structured) dtype, returns dtype id. |
| `nd.register_binary_op(dtype, kind, device, callback)` | Register a binary operation callback for a specific dtype and device. `kind` is one of `nd.BINARY_OP_ADD`, `nd.BINARY_OP_SUB`, … |

---

## ⚙️ Performance Considerations

- **Views** are cheap: they copy only shape, strides, offset, and a handle to the underlying `Tensor`.
- **Strided copy** is optimized for CPU and GPU; non‑contiguous copies use a fallback iterative kernel but are still efficient.
- **GPU addition** uses a highly optimized CUDA kernel that respects arbitrary strides, achieving near‑peak memory bandwidth.
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