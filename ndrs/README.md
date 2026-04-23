# ndrs

**ndrs** is a NumPy‑like tensor library for Rust, providing multi‑dimensional array (tensor) operations with **optional GPU acceleration** via CUDA. It emphasizes **zero‑copy views**, efficient strided operations, and a flexible ownership model.

---

## ✨ Features

- **N‑dimensional tensors** – shape, strides, and byte‑level data storage.
- **View‑based operations** – slicing, broadcasting, transposing, and reshaping without copying data.
- **Efficient strided copy** – fast data movement between non‑contiguous layouts.
- **Thread‑local and thread‑safe variants**  
  - `Rc<RefCell<Tensor>>` for single‑threaded speed  
  - `Arc<ReentrantMutex<RefCell<Tensor>>>` for multi‑threading and Python bindings
- **GPU acceleration** – transparent CPU ↔ GPU transfer, CUDA kernels for element‑wise addition.
- **CUDA stream support** – asynchronous execution, events for timing and cross‑stream synchronization (`wait_event`).
- **Operator overloading** – `+` and `+=` for tensors with broadcastable shapes.
- **Python‑like slicing** – intuitive `s!` macro: `s![1..4:2, ..]`.
- **Broadcasting** – automatic shape expansion.
- **Custom data types** – register your own types with user‑defined addition operations.
- **NPY file I/O** – load and save tensors from/to NumPy `.npy` files (preserving shape, supports `f32`/`i32`).
- **Python bindings** – use ndrs from Python via PyO3 (optional).

---

## 🚀 Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ndrs = "0.1"
```

### Basic CPU usage with the `tensor!` macro

```rust
use ndrs::{Tensor, s, tensor};

fn main() -> Result<(), String> {
    // Create tensors using the convenient `tensor!` macro (supports negatives, floats)
    let a = tensor!([[1, -2], [3, 4]]);      // automatically i32
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

    let a = tensor!([[1.0, 2.0], [3.0, 4.0]]);
    let b = tensor!([[5.0, 6.0], [7.0, 8.0]]);

    let a_gpu = a.into_arc().as_view().to_stream(&stream1)?;
    let b_gpu = b.into_arc().as_view().to_stream(&stream1)?;
    let mut out_gpu = a_gpu.create_output()?;

    // Record event to measure kernel time
    let start = stream1.record()?;
    a_gpu.add(&b_gpu, &mut out_gpu)?;
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
    let t = tensor!([[1.0, 2.0], [3.0, 4.0]]).into_rc();
    save_npy(&t, "example.npy")?;
    let loaded = load_npy("example.npy")?;
    assert_eq!(loaded.0.borrow().to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
```

### Custom data types

You can register your own types and define addition for them:

```rust
use ndrs::{register_dtype, register_add_op, TypeInfo, DType, Device};
use std::sync::Arc;

const DTYPE_MY_TYPE: DType = 1000;

#[repr(C)]
#[derive(Clone, Copy)]
struct MyType { a: i32, b: i32 }

fn mytype_add_op(a: *const u8, b: *const u8, c: *mut u8, n: usize, _dev: Device, _stream: Option<*mut std::ffi::c_void>) {
    let a_slice = unsafe { std::slice::from_raw_parts(a as *const MyType, n) };
    let b_slice = unsafe { std::slice::from_raw_parts(b as *const MyType, n) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c as *mut MyType, n) };
    for i in 0..n {
        c_slice[i] = MyType { a: a_slice[i].a + b_slice[i].a, b: a_slice[i].b + b_slice[i].b };
    }
}

register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
register_add_op(DTYPE_MY_TYPE, Arc::new(mytype_add_op));
```

---

## 🧠 Core Concepts

### `Tensor`
The raw data container. It owns a contiguous byte buffer (either on CPU or GPU) and stores shape, strides, data type, and device information. **It does not implement operations directly** – use `TensorView` for that.

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
- `set_current_device(id)` – sets the default device for context creation.
- `get_device_count()` – returns number of CUDA‑capable devices.
- `get_stream()` / `set_stream()` – thread‑local current CUDA stream.

### GPU streams and events

- **Streams** allow asynchronous command submission. Use `Stream::new()` to create a custom non‑default stream.
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
  Enables `cuda::*` functions, `ArcTensorView` GPU transfers, and GPU‑accelerated addition.

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

The Python API mirrors the Rust API: slicing, broadcasting, and arithmetic operators are supported.

---

## ⚙️ Performance Considerations

- **Views** are cheap: they copy only shape, strides, offset, and a handle to the underlying `Tensor`.
- **Strided copy** is optimized for CPU and GPU; non‑contiguous copies use a fallback iterative kernel but are still efficient.
- **GPU addition** uses a highly optimized CUDA kernel that respects arbitrary strides, achieving near‑peak memory bandwidth.
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
- NPY I/O powered by [npyz](https://crates.io/crates/npyz).