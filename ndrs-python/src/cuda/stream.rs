use ndrs::cuda::Stream as CudaStreamInner;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "Stream")]
pub struct PyCudaStream {
    inner: CudaStreamInner,
}

#[pymethods]
impl PyCudaStream {
    #[new]
    fn new(device_id: Option<usize>) -> PyResult<Self> {
        let inner =
            CudaStreamInner::new(device_id).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCudaStream { inner })
    }

    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn wait_event(&self, event: &super::event::PyCudaEvent) -> PyResult<()> {
        self.inner
            .wait_event(&event.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn record_event(&self) -> PyResult<super::event::PyCudaEvent> {
        let event = self
            .inner
            .record()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(super::event::PyCudaEvent { inner: event })
    }

    fn query(&self) -> bool {
        true
    }

    fn __repr__(&self) -> String {
        format!("<cuda.Stream>")
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudaStream>()?;
    Ok(())
}
