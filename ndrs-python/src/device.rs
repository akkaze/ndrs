use ndrs::device;
use ndrs::stream::{Event as CudaEvent, Stream as CudaStream};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;

// ---------- Device ----------
#[pyclass(name = "Device", unsendable)]
#[derive(Clone)]
pub struct PyDevice {
    pub inner: device::Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(device_type: &str, index: Option<usize>) -> PyResult<Self> {
        let inner = match device_type {
            "cpu" => device::Device::CPU,
            "cuda" => device::Device::GPU(index.unwrap_or(0)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown device type")),
        };
        Ok(PyDevice { inner })
    }

    fn __repr__(&self) -> String {
        match self.inner {
            device::Device::CPU => "device(type='cpu')".to_string(),
            device::Device::GPU(id) => format!("device(type='cuda', index={})", id),
        }
    }

    fn __enter__(&self) -> PyResult<Self> {
        match self.inner {
            device::Device::GPU(id) => device::set_current_device(id),
            device::Device::CPU => device::set_current_device(0),
        }
        Ok(self.clone())
    }

    fn __exit__(&self, _exc_type: &PyType, _exc_val: &PyAny, _exc_tb: &PyAny) -> PyResult<()> {
        Ok(())
    }
}

// ---------- Stream ----------
#[pyclass(name = "Stream", unsendable)]
#[derive(Clone)]
pub struct PyStream {
    pub inner: Arc<CudaStream>,
}

#[pymethods]
impl PyStream {
    #[new]
    pub fn new() -> PyResult<Self> {
        let s = CudaStream::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyStream { inner: Arc::new(s) })
    }

    pub fn synchronize(&self) -> PyResult<()> {
        self.inner.synchronize().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn wait_event(&self, event: &PyEvent) -> PyResult<()> {
        self.inner.wait_event(&event.inner).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn join(&self, other: &PyStream) -> PyResult<()> {
        self.inner.join(&other.inner).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn record(&self) -> PyResult<PyEvent> {
        let event = self.inner.record().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyEvent { inner: event })
    }

    fn __enter__(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    fn __exit__(&self, _exc_type: &PyType, _exc_val: &PyAny, _exc_tb: &PyAny) -> PyResult<()> {
        Ok(())
    }
}

// ---------- Event ----------
#[pyclass(name = "Event", unsendable)]
pub struct PyEvent {
    pub inner: CudaEvent,
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
        self.inner.synchronize().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }
}

// ---------- Module functions ----------
#[pyfunction]
pub fn is_cuda_available() -> bool {
    device::get_cuda_device_count().unwrap_or(0) > 0
}

#[pyfunction]
pub fn get_cuda_device_count() -> PyResult<usize> {
    device::get_cuda_device_count().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn null_stream() -> PyResult<PyStream> {
    PyStream::new()
}

#[pyfunction]
pub fn get_current_device() -> PyResult<String> {
    match device::get_current_device() {
        Some(id) => Ok(format!("cuda:{}", id)),
        None => Ok("cpu".to_string()),
    }
}