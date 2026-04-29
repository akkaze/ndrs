pub mod device;
pub mod elementwise_kernel;
pub mod event;
pub mod raw_kernel;
pub mod stream;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    device::register(m)?;
    event::register(m)?;
    stream::register(m)?;
    elementwise_kernel::register(m)?;
    Ok(())
}
