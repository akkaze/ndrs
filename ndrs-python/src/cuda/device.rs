use ndrs::cuda;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    Ok(())
}

#[pyfunction]
fn set_device(device_id: usize) -> PyResult<()> {
    cuda::set_device(device_id).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn get_device() -> usize {
    cuda::get_device()
}

#[pyfunction]
fn get_device_count() -> PyResult<usize> {
    cuda::get_device_count().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn is_available() -> bool {
    cuda::is_available()
}
