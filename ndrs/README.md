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
- **GPU acceleration** – transparent CPU ↔ GPU transfer, CUDA kernels for strided element‑wise addition.
- **Operator overloading** – `+` and `+=` for tensors with broadcastable shapes.
- **Python‑like slicing** – intuitive `s!` macro: `s![1..4:2, ..]`.
- **Broadcasting** – automatic shape expansion for arithmetic.
- **Python bindings** – use ndrs from Python via PyO3 (optional).

---

## 🚀 Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ndrs = "0.1"
```

### Basic CPU usage

```rust
use ndrs::{Tensor, s};

fn main() -> Result<(), String> {
    // Create tensors from `Vec<f32>` and shape
    let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Wrap into a thread‑local shared view (Rc<RefCell<Tensor>>)
    let a_view = a.into_rc().as_view();
    let b_view = b.into_rc().as_view();

    // Element‑wise addition (requires broadcastable shapes)
    let c_view = a_view + b_view;
    assert_eq!(c_view.shape(), &[2, 2]);

    // Slice and assign
    let mut a_mut = a_view.clone();
    let mut sub = a_mut.slice(&s![0..1, ..])?;
    let b_sub = b_view.slice(&s![0..1, ..])?;
    sub.assign(&b_sub)?;

    // Convert back to a `Vec<f32>`
    let result = c_view.to_vec::<f32>()?;
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

    Ok(())
}
```

### GPU usage (CUDA)

```rust
use ndrs::{Tensor, Device};

fn gpu_add_example() -> Result<(), String> {
    // Create CPU tensors
    let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Wrap into thread‑safe views (Arc<ReentrantMutex<RefCell<Tensor>>>)
    let a_view = a.into_arc().as_view();
    let b_view = b.into_arc().as_view();

    // Transfer to GPU (device 0)
    let a_gpu = a_view.to_gpu(0)?;
    let b_gpu = b_view.to_gpu(0)?;

    // Create an output tensor on GPU (automatically allocated)
    let out_gpu = a_gpu.create_output()?;   // shape [2,2], zeros
    let mut out_gpu = a_gpu.add(&b_gpu, &out_gpu)?;   // in‑place not supported, returns new view

    // Bring result back to CPU
    let out_cpu = out_gpu.to_cpu()?;
    let result = out_cpu.to_vec::<f32>()?;
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

    Ok(())
}
```

> **Note:** GPU operations require the `cuda` feature and a CUDA‑capable device.

---

## 🧠 Core Concepts

### `Tensor`
The raw data container. It owns a contiguous byte buffer (either on CPU or GPU) and stores shape, strides, data type, and device information. **It does not implement operations directly** – use `TensorView` for that.

### `TensorView`
A view into a `Tensor` with an optional offset, shape, and strides. All mathematical operations (addition, slicing, broadcasting, device transfer) are defined on views.

- `RcTensorView` – thread‑local variant using `Rc<RefCell<Tensor>>`. Fast and lightweight for single‑threaded code.
- `ArcTensorView` – thread‑safe variant using `Arc<ReentrantMutex<RefCell<Tensor>>>`. Required for multi‑threaded environments and Python bindings.

### Slice macro `s!`
Creates a slice descriptor for the `.slice()` method. Supports ranges, steps, single indices, and `..` (all).

```rust
let sub = view.slice(&s![1..4, 2..6])?;      // rows 1..4, cols 2..6
let row = view.slice(&s![2, ..])?;           // single row (dimension reduced)
let col = view.slice(&s![.., 3])?;           // single column
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

---

## 📦 Cargo Features

- **default** – CPU only.
- **cuda** – enables GPU support (requires CUDA toolkit and `cudarc`).

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
print(t3.numpy())   # [[4. 6.]
                    #  [8. 10.]]
```

The Python API mirrors the Rust API: slicing, broadcasting, and arithmetic operators are supported.

---

## ⚙️ Custom Data Types

You can register your own data types for tensor operations:

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
        c_slice[i].a = a_slice[i].a + b_slice[i].a;
    }
}

register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
register_add_op(DTYPE_MY_TYPE, Arc::new(mytype_add_op));
```

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
```