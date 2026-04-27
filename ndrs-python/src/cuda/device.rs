use ndrs::cuda;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use ndrs::Device;
use std::str::FromStr;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    Ok(())
}

#[pyfunction]
fn get_device() -> PyResult<String> {
    let idx = cuda::get_device();
    let device = Device::Cuda(idx);
    Ok(device.to_string())
}

#[pyfunction]
fn set_device(device_str: &str) -> PyResult<()> {
    let device = Device::from_str(device_str)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    match device {
        Device::Cuda(idx) => cuda::set_device(idx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        Device::Cpu => Err(PyRuntimeError::new_err("Cannot set device to CPU using this function; use CUDA device string like 'cuda:0'")),
    }
}

#[pyfunction]
fn get_device_count() -> PyResult<usize> {
    cuda::get_device_count().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn is_available() -> bool {
    cuda::is_available()
}
