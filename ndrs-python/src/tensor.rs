use ndrs::tensor::ArcTensor;
use ndrs::{DType, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn parse_device(s: &str) -> Result<Device, String> {
    s.parse()
}

#[pyclass(name = "PyTensor")]
pub struct PyTensor {
    inner: ArcTensor,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    fn from_bytes(
        py: Python,
        bytes: &Bound<'_, PyBytes>,
        shape: Vec<usize>,
        dtype_id: u32, // 改为 u32
        device: Option<String>,
    ) -> PyResult<Self> {
        let dev = device.unwrap_or_else(|| "cpu".to_string());
        let dev = parse_device(&dev).map_err(|e| PyRuntimeError::new_err(e))?;
        let tensor = Tensor::new_from_bytes(
            bytes.as_bytes().to_vec().into_boxed_slice(),
            shape,
            dtype_id,
            dev,
        )
        .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(PyTensor {
            inner: tensor.into_arc(),
        })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    fn dtype(&self) -> String {
        match self.inner.dtype() {
            DTYPE_FLOAT32 => "float32".to_string(),
            DTYPE_INT32 => "int32".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    fn __add__(&self, other: &PyTensor) -> Self {
        PyTensor {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn as_bytes<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyBytes>> {
        let cpu_view = self
            .inner
            .as_view()
            .to_cpu()
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let guard = cpu_view.handle().0.lock();
        let tensor = guard.borrow();
        let bytes = tensor
            .as_bytes()
            .ok_or_else(|| PyRuntimeError::new_err("No bytes"))?;
        Ok(PyBytes::new(py, bytes))
    }

    fn dtype_id(&self) -> u32 {
        self.inner.dtype()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
