# ndrs

**ndrs** is a NumPy-like tensor library for Rust, providing multi-dimensional array (tensor) operations with optional GPU acceleration via CUDA.

## Features

- **N-dimensional tensors** with shape and strides
- **View-based operations** (slicing, broadcasting, transposing) without copying data
- **Efficient strided copy** for reshaping and indexing
- **Thread‑local** (`Rc<RefCell<Tensor>>`) and **thread‑safe** (`Arc<Mutex<Tensor>>`) variants
- **GPU support** – transparent transfer between CPU and GPU, CUDA kernels for element‑wise addition
- **Operator overloading** – `+`, `+=` for tensors (same shape)
- **Slice macro** `s!` – intuitive Python‑like slicing syntax (e.g., `s![1..4:2, ..]`)
- **Python bindings** (via PyO3) – use `ndrs` from Python

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ndrs = "0.1"
```

### Basic Usage (CPU)

```rust
use ndrs::{Tensor, RcTensorView, s};

fn main() -> Result<(), String> {
    // Create tensors from slices
    let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // Convert to shared view (thread‑local variant)
    let a_view = a.into_rc().as_view();
    let b_view = b.into_rc().as_view();

    // Add
    let c_view = a_view + b_view;
    assert_eq!(c_view.shape(), &[2, 2]);

    // Slice and assign
    let mut a_mut = a_view.clone();
    let mut sub = a_mut.slice(&s![0..1, ..])?;
    sub.assign(&b_view.slice(&s![0..1, ..])?)?;

    Ok(())
}
```

### GPU Usage (CUDA)

```rust
use ndrs::{Tensor, ArcTensorView, Device};

let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);

// Wrap in Arc<Mutex<Tensor>> for thread safety
let a_arc = a.into_arc();
let b_arc = b.into_arc();

let a_view = a_arc.as_view();
let b_view = b_arc.as_view();

// Transfer to GPU (device 0)
let mut a_gpu = a_view.clone();
let mut b_gpu = b_view.clone();
a_view.to_gpu(&mut a_gpu, 0)?;
b_view.to_gpu(&mut b_gpu, 0)?;

// Output on GPU
let out_tensor = Tensor::new_cpu_from_bytes(
    vec![0u8; 2 * std::mem::size_of::<f32>()].into_boxed_slice(),
    vec![2],
    a_view.dtype(),
)?;
let out_handle = out_tensor.into_arc();
let mut out_view = out_handle.as_view();
let mut out_gpu = out_view.clone();
out_view.to_gpu(&mut out_gpu, 0)?;

// Add on GPU
ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu)?;

// Bring result back to CPU
let mut out_cpu = out_gpu.clone();
out_gpu.to_cpu(&mut out_cpu)?;
// ... use out_cpu
```

## Core Concepts

- **`Tensor`** – raw data container (CPU or GPU), does **not** implement operations directly.
- **`TensorView`** – a view into a `Tensor` with optional offset, shape, and strides. All operations are defined on views.
- **`RcTensorView`** – thread‑local variant using `Rc<RefCell<Tensor>>`. Lightweight and fast for single‑threaded code.
- **`ArcTensorView`** – thread‑safe variant using `Arc<Mutex<Tensor>>`. Required for Python bindings and multi‑threading.
- **`s!` macro** – creates a slice descriptor: `s![start..end:step, ..]`. Supports ranges, steps, single indices, and `..` (all).

## Slicing Examples

```rust
let view = ...;
let sub = view.slice(&s![1..4, 2..6])?;        // rows 1..4, cols 2..6
let row = view.slice(&s![2, ..])?;             // single row (dimension reduced)
let col = view.slice(&s![.., 3])?;             // single column
let every_other = view.slice(&s![0..8:2, ..])?; // every second row
```

## Broadcasting

Broadcasting is supported via `broadcast_to` method and `broadcast_shapes` helper.

```rust
let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
let b = Tensor::new_cpu_from_f32(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
let target = broadcast_shapes(a.shape(), b.shape()).unwrap(); // [3,4]
let a_bcast = a_view.broadcast_to(&target)?;
let b_bcast = b_view.broadcast_to(&target)?;
```

## Device Management

- `Device::CPU` – operations on host memory.
- `Device::GPU(id)` – operations on CUDA device `id`.
- `set_current_device(id)` – sets the default device for context creation.
- `get_cuda_device_count()` – returns number of CUDA‑capable devices.

## GPU Kernel Support

Currently, `ndrs` includes CUDA kernels for:
- Element‑wise addition (`f32` and `i32`)
- Strided copy (byte‑wise)
- Contiguous conversion

These are exposed via the kernel module and used internally by view operations.

## Python Bindings

The `ndrs-python` crate (not yet published) provides Python bindings. Usage:

```python
import ndrs as nd
t = nd.Tensor([[1,2],[3,4]], dtype=nd.float32)
t2 = t.to("cuda")
t3 = t2 + t2
print(t3.numpy())
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/yourusername/ndrs).

## Acknowledgments

- Inspired by NumPy, PyTorch, and `ndarray` crate.
- Uses `cudarc` for CUDA bindings and `bytemuck` for safe byte casts.