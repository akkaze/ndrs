use pyo3::prelude::*;

mod cuda;
mod tensor;

#[pymodule]
fn ndrs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 cuda 子模块
    let cuda_module = PyModule::new(m.py(), "cuda")?;
    cuda::device::register(&cuda_module)?;
    cuda::stream::register(&cuda_module)?;
    cuda::event::register(&cuda_module)?;
    m.add_submodule(&cuda_module)?;

    // 注册 tensor 模块
    tensor::register(m)?;

    Ok(())
}
