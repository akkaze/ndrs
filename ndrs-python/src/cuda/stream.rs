use ndrs::cuda::Stream as CudaStreamInner;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use ndrs::cuda;

#[pyclass(name = "Stream")]
#[derive(Clone)]
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

#[pyfunction]
fn get_stream() -> PyResult<PyCudaStream> {
    let stream = cuda::get_stream()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyCudaStream { inner: stream })
}

#[pyfunction]
fn set_stream(stream: PyCudaStream) -> PyResult<()> {
    cuda::set_stream(stream.inner)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudaStream>()?;
    m.add_function(wrap_pyfunction!(get_stream, m)?)?;
    m.add_function(wrap_pyfunction!(set_stream, m)?)?;
    Ok(())
}
