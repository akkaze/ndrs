use crate::cuda::stream::PyCudaStream;
use crate::view::_TensorView;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndrs::cuda::{get_stream, RawKernel};
use ndrs::view::TensorViewOps;
use ndrs::ArcTensorView;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

#[pyclass(name = "RawKernel")]
pub struct PyRawKernel {
    pub(crate) inner: Arc<RawKernel>,
}

enum Arg {
    TensorReadWrite(ArcTensorView),
    CudaSliceF32(CudaSlice<f32>),
    CudaSliceI32(CudaSlice<i32>),
    I32(i32),
    F32(f32),
    Usize(usize),
    BoolAsI32(i32),
}
#[pymethods]
impl PyRawKernel {
    fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: Bound<PyList>,
        py: Python<'_>,
        stream: Option<&PyCudaStream>,
    ) -> PyResult<()> {
        let stream = match stream {
            Some(s) => s.inner.clone(),
            None => get_stream().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };

        let mut collected: Vec<Arg> = Vec::new();

        for arg_ref in args.iter() {
            if let Ok(bound_view) = arg_ref.cast::<_TensorView>() {
                let view = bound_view.borrow();
                let inner = view.inner.clone();
                collected.push(Arg::TensorReadWrite(inner));
                continue;
            }

            // 标量
            if let Ok(val) = arg_ref.extract::<i32>() {
                collected.push(Arg::I32(val));
                continue;
            }
            if let Ok(val) = arg_ref.extract::<f32>() {
                collected.push(Arg::F32(val));
                continue;
            }
            if let Ok(val) = arg_ref.extract::<usize>() {
                collected.push(Arg::Usize(val));
                continue;
            }
            if let Ok(val) = arg_ref.extract::<bool>() {
                collected.push(Arg::BoolAsI32(val as i32));
                continue;
            }

            if let Ok(list) = arg_ref.downcast::<PyList>() {
                // 尝试转换为 f32 并分配 GPU 内存
                if let Ok(vec) = list.extract::<Vec<f32>>() {
                    let slice = stream
                        .inner()
                        .clone_htod(&vec)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    collected.push(Arg::CudaSliceF32(slice));
                    continue;
                }
                // 尝试转换为 i32
                if let Ok(vec) = list.extract::<Vec<i32>>() {
                    let slice = stream
                        .inner()
                        .clone_htod(&vec)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    collected.push(Arg::CudaSliceI32(slice));
                    continue;
                }
                return Err(PyTypeError::new_err(
                    "List elements must be all f32 or all i32",
                ));
            }

            return Err(PyTypeError::new_err("Unsupported argument type"));
        }

        let mut builder = self.inner.launch_builder(stream.inner());

        for arg in &mut collected {
            match arg {
                Arg::TensorReadWrite(inner) => {
                    let slice = unsafe { inner.as_gpu_slice_mut() };
                    builder.arg(slice);
                }
                Arg::I32(v) => {
                    builder.arg(&*v);
                }
                Arg::F32(v) => {
                    builder.arg(&*v);
                }
                Arg::Usize(v) => {
                    builder.arg(&*v);
                }
                Arg::BoolAsI32(v) => {
                    builder.arg(&*v);
                }
                Arg::CudaSliceF32(slice) => {
                    builder.arg(slice); // &CudaSlice<f32> 实现了 PushKernelArg
                }
                Arg::CudaSliceI32(slice) => {
                    builder.arg(slice);
                }
            }
        }

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        stream
            .inner()
            .synchronize()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}
