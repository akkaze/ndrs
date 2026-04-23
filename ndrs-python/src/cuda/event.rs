use ndrs::cuda::Event as CudaEventInner;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "Event")]
pub struct PyCudaEvent {
    pub(crate) inner: CudaEventInner,
}

#[pymethods]
impl PyCudaEvent {
    #[new]
    fn new(device_id: usize, _enable_timing: bool, _blocking: bool) -> PyResult<Self> {
        let inner =
            CudaEventInner::new(device_id).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCudaEvent { inner })
    }

    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn wait(&self, stream: &super::stream::PyCudaStream) -> PyResult<()> {
        stream.wait_event(self)
    }

    fn elapsed_time(&self, end_event: &PyCudaEvent) -> PyResult<f64> {
        let dur = end_event
            .inner
            .elapsed_since(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(dur.as_secs_f64() * 1000.0)
    }

    fn __repr__(&self) -> String {
        format!("<cuda.Event>")
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudaEvent>()?;
    Ok(())
}
