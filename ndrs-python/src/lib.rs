use ndrs::{BinaryOpKind, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::prelude::*;

mod register;
mod tensor;

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
fn _ndrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 tensor 类
    tensor::register(m)?;
    // 添加注册函数
    m.add_function(wrap_pyfunction!(register::register_dtype_py, m)?)?;
    m.add_function(wrap_pyfunction!(register::register_binary_op_raw, m)?)?;
    // 添加常量
    register_constants(m)?;
    Ok(())
}
