use ndrs::device;
use ndrs::stream::{Event as CudaEvent, Stream as CudaStream};
use ndrs::{
    broadcast_shapes, ArcTensorView, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32,
};
use numpy::{PyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

// ---------- Device ----------
#[pyclass(name = "Device", unsendable)]
#[derive(Clone)]
pub struct PyDevice {
    pub inner: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(device_type: &str, index: Option<usize>) -> PyResult<Self> {
        let device = match device_type {
            "cpu" => Device::CPU,
            "cuda" => Device::GPU(index.unwrap_or(0)),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unknown device type",
                ))
            }
        };
        Ok(PyDevice { inner: device })
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Device::CPU => "device(type='cpu')".to_string(),
            Device::GPU(id) => format!("device(type='cuda', index={})", id),
        }
    }

    fn __enter__(&self) -> PyResult<Self> {
        match self.inner {
            Device::GPU(id) => device::set_current_device(id),
            Device::CPU => device::set_current_device(0),
        }
        Ok(self.clone())
    }

    fn __exit__(&self, _exc_type: &PyAny, _exc_val: &PyAny, _exc_tb: &PyAny) -> PyResult<()> {
        Ok(())
    }
}

// ---------- Stream ----------
#[pyclass(name = "Stream", unsendable)]
#[derive(Clone)]
struct PyStream {
    inner: Arc<CudaStream>,
}

#[pymethods]
impl PyStream {
    #[new]
    fn new() -> PyResult<Self> {
        let s = CudaStream::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyStream { inner: Arc::new(s) })
    }

    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    fn wait_event(&self, event: &PyEvent) -> PyResult<()> {
        self.inner
            .wait_event(&event.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    fn join(&self, other: &PyStream) -> PyResult<()> {
        self.inner
            .join(&other.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    fn record(&self) -> PyResult<PyEvent> {
        let event = self
            .inner
            .record()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyEvent { inner: event })
    }

    fn __enter__(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    fn __exit__(&self, _exc_type: &PyAny, _exc_val: &PyAny, _exc_tb: &PyAny) -> PyResult<()> {
        Ok(())
    }
}

// ---------- Event ----------
#[pyclass(name = "Event", unsendable)]
struct PyEvent {
    inner: CudaEvent,
}

#[pymethods]
impl PyEvent {
    #[new]
    fn new() -> PyResult<Self> {
        let dev_id = device::get_current_device()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No current device"))?;
        let ctx = device::get_or_create_context(dev_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let event = CudaEvent::new_with_context(&ctx)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyEvent { inner: event })
    }

    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }
}

// ---------- Tensor ----------
#[pyclass(name = "Tensor", unsendable)]
pub struct PyTensor {
    inner: ArcTensorView,
}

impl PyTensor {
    fn from_view(view: ArcTensorView) -> Self {
        PyTensor { inner: view }
    }
}

// 辅助函数：将 Python 嵌套列表转换为扁平 f32 向量和形状
fn flatten_list_f32(obj: &PyAny, shape: &mut Vec<usize>) -> PyResult<Vec<f32>> {
    if let Ok(list) = obj.extract::<&PyList>() {
        shape.push(list.len());
        let mut result = Vec::new();
        for item in list.iter() {
            result.extend(flatten_list_f32(&item, &mut Vec::new())?);
        }
        Ok(result)
    } else {
        let val: f64 = obj.extract()?;
        Ok(vec![val as f32])
    }
}

// 辅助函数：将 Python 嵌套列表转换为扁平 i32 向量和形状
#[allow(dead_code)]
fn flatten_list_i32(obj: &PyAny, shape: &mut Vec<usize>) -> PyResult<Vec<i32>> {
    if let Ok(list) = obj.extract::<&PyList>() {
        shape.push(list.len());
        let mut result = Vec::new();
        for item in list.iter() {
            result.extend(flatten_list_i32(&item, &mut Vec::new())?);
        }
        Ok(result)
    } else {
        let val: i64 = obj.extract()?;
        Ok(vec![val as i32])
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, dtype=None, _device=None))]
    fn new(data: &PyAny, dtype: Option<u32>, _device: Option<PyDevice>) -> PyResult<Self> {
        let target_dtype = if let Some(d) = dtype {
            d
        } else {
            let is_int = if let Ok(list) = data.extract::<&PyList>() {
                list.iter().all(|item| item.extract::<i64>().is_ok())
            } else if let Ok(_arr) = data.extract::<&PyArrayDyn<i32>>() {
                true
            } else {
                false
            };
            if is_int {
                DTYPE_INT32
            } else {
                DTYPE_FLOAT32
            }
        };
        match target_dtype {
            DTYPE_FLOAT32 => {
                let (data_f32, shape) = if let Ok(arr) = data.extract::<&PyArrayDyn<f32>>() {
                    (
                        arr.readonly().as_slice().unwrap().to_vec(),
                        arr.shape().iter().map(|&s| s as usize).collect(),
                    )
                } else {
                    let mut shape = Vec::new();
                    let data = flatten_list_f32(data, &mut shape)?;
                    (data, shape)
                };
                let tensor = Tensor::new_cpu_from_f32(data_f32, shape);
                let view = ArcTensorView::new(tensor.into_arc());
                Ok(PyTensor::from_view(view))
            }
            DTYPE_INT32 => {
                let data_i32: Vec<i32> = if let Ok(arr) = data.extract::<&PyArrayDyn<i32>>() {
                    arr.readonly().as_slice().unwrap().to_vec()
                } else if let Ok(list) = data.extract::<&PyList>() {
                    list.extract::<Vec<i64>>()?
                        .iter()
                        .map(|&x| x as i32)
                        .collect()
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert data to i32",
                    ));
                };
                let shape = vec![data_i32.len()];
                let tensor = Tensor::new_cpu_from_i32(data_i32, shape);
                let view = ArcTensorView::new(tensor.into_arc());
                Ok(PyTensor::from_view(view))
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Unsupported dtype")),
        }
    }

    #[staticmethod]
    fn from_numpy(arr: &PyAny, _dtype: Option<u32>, _device: Option<PyDevice>) -> PyResult<Self> {
        if let Ok(arr_f32) = arr.extract::<&PyArrayDyn<f32>>() {
            let shape = arr_f32.shape().iter().map(|&s| s as usize).collect();
            let vec = arr_f32.readonly().as_slice().unwrap().to_vec();
            let tensor = Tensor::new_cpu_from_f32(vec, shape);
            let view = ArcTensorView::new(tensor.into_arc());
            return Ok(PyTensor::from_view(view));
        }
        if let Ok(arr_i32) = arr.extract::<&PyArrayDyn<i32>>() {
            let shape = arr_i32.shape().iter().map(|&s| s as usize).collect();
            let vec = arr_i32.readonly().as_slice().unwrap().to_vec();
            let tensor = Tensor::new_cpu_from_i32(vec, shape);
            let view = ArcTensorView::new(tensor.into_arc());
            return Ok(PyTensor::from_view(view));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported numpy array dtype, only float32 and int32 are supported",
        ))
    }

    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        let view = &self.inner;
        let elem_size = ndrs::dtype::get_dtype_info(view.dtype()).unwrap().size;
        let total_bytes = view.shape().iter().product::<usize>() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            view.shape().to_vec(),
            view.dtype(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let out_handle = out_tensor.into_arc();
        let mut out_view = ArcTensorView::new(out_handle);
        view.contiguous(&mut out_view)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let tensor_contig = out_view.into_handle(); // 获取内部的 Arc<Mutex<Tensor>>
        let tensor_contig_ref = tensor_contig.lock().unwrap();
        let bytes = tensor_contig_ref
            .as_bytes()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Cannot get bytes"))?;
        let shape = tensor_contig_ref.shape();

        match tensor_contig_ref.dtype() {
            DTYPE_FLOAT32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const f32,
                        tensor_contig_ref.size(),
                    )
                };
                let array = PyArray::from_vec(py, slice.to_vec())
                    .reshape(shape)
                    .map_err(|e: PyErr| e)?;
                Ok(array.into())
            }
            DTYPE_INT32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const i32,
                        tensor_contig_ref.size(),
                    )
                };
                let array = PyArray::from_vec(py, slice.to_vec())
                    .reshape(shape)
                    .map_err(|e: PyErr| e)?;
                Ok(array.into())
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported dtype for numpy conversion",
            )),
        }
    }

    fn to(&self, device: &str) -> PyResult<Self> {
        let target = match device {
            "cpu" => Device::CPU,
            "cuda" => Device::GPU(0),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown device")),
        };
        let src_view = &self.inner;
        let elem_size = ndrs::dtype::get_dtype_info(src_view.dtype()).unwrap().size;
        let total_bytes = src_view.shape().iter().product::<usize>() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            src_view.shape().to_vec(),
            src_view.dtype(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let out_handle = out_tensor.into_arc();
        let mut out_view = ArcTensorView::new(out_handle);
        src_view
            .to(&mut out_view, target)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyTensor::from_view(out_view))
    }

    fn view(&self) -> PyResult<PyTensorView> {
        Ok(PyTensorView {
            inner: self.inner.clone(),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> u32 {
        self.inner.dtype()
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice {
            inner: self.inner.handle().lock().unwrap().device(),
        }
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let a_view = &self.inner;
        let b_view = &other.inner;
        let target_shape = broadcast_shapes(a_view.shape(), b_view.shape()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Incompatible shapes for broadcast")
        })?;
        let a_bcast = a_view
            .broadcast_to(&target_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let b_bcast = b_view
            .broadcast_to(&target_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let elem_size = ndrs::dtype::get_dtype_info(a_view.dtype()).unwrap().size;
        let total_bytes = target_shape.iter().product::<usize>() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            target_shape,
            a_view.dtype(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let out_handle = out_tensor.into_arc();
        let mut out_view = ArcTensorView::new(out_handle);
        ArcTensorView::add(&a_bcast, &b_bcast, &mut out_view)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyTensor::from_view(out_view))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={:?})",
            self.inner.shape(),
            self.inner.dtype(),
            self.inner.handle().lock().unwrap().device()
        )
    }
}

// ---------- TensorView ----------
#[pyclass(name = "TensorView", unsendable)]
pub struct PyTensorView {
    inner: ArcTensorView,
}

#[pymethods]
impl PyTensorView {
    fn as_strided(&self, new_shape: Vec<usize>, new_strides: Vec<usize>, offset: usize) -> Self {
        PyTensorView {
            inner: self.inner.as_strided(new_shape, new_strides, offset),
        }
    }

    fn broadcast_to(&self, target_shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self
                .inner
                .broadcast_to(&target_shape)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }

    fn transpose(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self
                .inner
                .transpose(&axes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }
    fn T(&self) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self
                .inner
                .T()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }

    fn contiguous(&mut self, out: &mut PyTensorView) -> PyResult<()> {
        self.inner
            .contiguous(&mut out.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    fn assign(&mut self, src: &PyTensorView) -> PyResult<()> {
        self.inner
            .assign(&src.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

// ---------- Module functions ----------
#[pyfunction]
fn is_cuda_available() -> bool {
    device::get_cuda_device_count().unwrap_or(0) > 0
}

#[pyfunction]
fn get_cuda_device_count() -> PyResult<usize> {
    device::get_cuda_device_count()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn null_stream() -> PyResult<PyStream> {
    PyStream::new()
}

#[pyfunction]
fn get_current_device() -> PyResult<String> {
    match device::get_current_device() {
        Some(id) => Ok(format!("cuda:{}", id)),
        None => Ok("cpu".to_string()),
    }
}

// ---------- Module ----------
#[pymodule]
fn _ndrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorView>()?;
    m.add_class::<PyStream>()?;
    m.add_class::<PyEvent>()?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_cuda_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(null_stream, m)?)?;
    m.add_function(wrap_pyfunction!(get_current_device, m)?)?;
    m.add("float32", DTYPE_FLOAT32)?;
    m.add("int32", DTYPE_INT32)?;
    Ok(())
}
