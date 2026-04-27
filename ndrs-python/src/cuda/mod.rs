pub mod device;
pub mod event;
pub mod stream;
use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    device::register(m)?;
    event::register(m)?;
    stream::register(m)?;
    Ok(())
}