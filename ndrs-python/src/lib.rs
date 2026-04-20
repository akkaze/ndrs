use pyo3::prelude::*;

mod device;
mod tensor;
mod ops;
mod indexing;
mod conversion;
mod utils;

#[pymodule]
fn _ndrs(_py: Python, m: &PyModule) -> PyResult<()> {
    // 注册类
    m.add_class::<device::PyDevice>()?;
    m.add_class::<device::PyStream>()?;
    m.add_class::<device::PyEvent>()?;
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tensor::PyTensorView>()?;

    // 注册函数
    m.add_function(wrap_pyfunction!(device::is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(device::get_cuda_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(device::null_stream, m)?)?;
    m.add_function(wrap_pyfunction!(device::get_current_device, m)?)?;

    // 常量
    m.add("float32", ndrs::dtype::DTYPE_FLOAT32)?;
    m.add("int32", ndrs::dtype::DTYPE_INT32)?;

    Ok(())
}