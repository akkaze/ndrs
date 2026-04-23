use crate::tensor::PyTensor;
use ndrs::{broadcast_shapes, ArcTensorView, Tensor, TensorViewOps};
use pyo3::prelude::*;

pub fn tensor_add_impl(a: &ArcTensorView, b: &ArcTensorView) -> PyResult<PyTensor> {
    let target_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Incompatible shapes for broadcast")
    })?;
    let a_bcast = a
        .broadcast_to(&target_shape)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let b_bcast = b
        .broadcast_to(&target_shape)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let elem_size = ndrs::dtype::get_dtype_info(a.dtype()).unwrap().size;
    let total_bytes = target_shape.iter().product::<usize>() * elem_size;
    let out_tensor = Tensor::new_cpu_from_bytes(
        vec![0u8; total_bytes].into_boxed_slice(),
        target_shape,
        a.dtype(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let out_handle = out_tensor.into_arc();
    let mut out_view = ArcTensorView::new(out_handle);
    ArcTensorView::add(&a_bcast, &b_bcast, &mut out_view)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(PyTensor::from_view(out_view))
}
