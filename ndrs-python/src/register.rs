use ndrs::{register_binary_op, BinaryOpFn, BinaryOpKind, Device, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::Arc;

#[pyfunction]
pub fn register_dtype_py(name: String, size: usize) -> PyResult<u32> {
    let dtype = ndrs::allocate_dtype();
    ndrs::register_dtype(dtype, ndrs::TypeInfo { size, name });
    Ok(dtype)
}

#[pyfunction]
pub fn register_binary_op_raw(
    dtype: u32,
    kind: u32,
    device_str: &str,
    fn_ptr: usize, // 函数地址作为 usize
) -> PyResult<()> {
    let device = match device_str {
        "cpu" => Device::Cpu,
        s if s.starts_with("cuda:") => {
            let idx = s[5..]
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid CUDA device"))?;
            Device::Cuda(idx)
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown device")),
    };
    let kind_enum = match kind {
        0 => BinaryOpKind::Add,
        1 => BinaryOpKind::Sub,
        2 => BinaryOpKind::Mul,
        3 => BinaryOpKind::Div,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid op kind")),
    };

    // 将 usize 转换为函数指针类型
    type RawBinaryOp = extern "C" fn(*const u8, *const u8, *mut u8, usize, i32, *mut c_void) -> i32;
    let op: BinaryOpFn = Arc::new(
        move |a, _a_strides, b, _b_strides, c, _c_strides, _shape, _ndim, n, dev, stream_opt| {
            let stream_ptr = stream_opt.unwrap_or(std::ptr::null_mut());
            let dev_code = match dev {
                Device::Cpu => 0,
                Device::Cuda(idx) => (idx + 1) as i32,
            };
            // 安全：fn_ptr 来自 Python 端的有效函数指针
            let f: RawBinaryOp = unsafe { std::mem::transmute(fn_ptr) };
            let ret = f(a, b, c, n, dev_code, stream_ptr);
            if ret == 0 {
                Ok(())
            } else {
                Err(format!("Custom op failed with code {}", ret))
            }
        },
    );

    register_binary_op(dtype, kind_enum, device, op).map_err(|e| PyRuntimeError::new_err(e))
}

#[pyfunction]
pub fn allocate_dtype_py() -> PyResult<u32> {
    Ok(ndrs::allocate_dtype())
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_dtype_py, m)?)?;
    m.add_function(wrap_pyfunction!(register_binary_op_raw, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_dtype_py, m)?)?;
    Ok(())
}
