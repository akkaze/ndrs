use ndrs::{BinaryOpKind, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::prelude::*;

mod register;
mod tensor;
mod view;
mod cuda;

use tensor::PyTensor;
use view::PyTensorView;

fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DTYPE_FLOAT32", DTYPE_FLOAT32)?;
    m.add("DTYPE_INT32", DTYPE_INT32)?;
    m.add("BINARY_OP_ADD", BinaryOpKind::Add.as_u32())?;
    m.add("BINARY_OP_SUB", BinaryOpKind::Sub.as_u32())?;
    m.add("BINARY_OP_MUL", BinaryOpKind::Mul.as_u32())?;
    m.add("BINARY_OP_DIV", BinaryOpKind::Div.as_u32())?;
    Ok(())
}

#[pymodule]
fn _ndrs(py: Python,m: &Bound<'_, PyModule>) -> PyResult<()> {
    tensor::register(m)?;
    view::register(m)?;
    let cuda_mod = PyModule::new(py, "cuda")?;
    cuda::register(&cuda_mod)?;
    m.add_submodule(&cuda_mod)?;
    // 添加注册函数
    m.add_function(wrap_pyfunction!(register::register_dtype_py, m)?)?;
    m.add_function(wrap_pyfunction!(register::register_binary_op_raw, m)?)?;
    // 添加常量
    register_constants(m)?;
    Ok(())
}
