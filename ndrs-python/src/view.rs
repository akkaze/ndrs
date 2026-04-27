use crate::PyTensor;
use ndrs::tensor::ArcTensor;
use ndrs::ArcTensorView;
use ndrs::{DType, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "PyTensorView")]
pub struct PyTensorView {
    pub(crate) inner: ArcTensorView,
}

#[pymethods]
impl PyTensorView {
    // fn shape(&self) -> Vec<usize> {
    //     self.inner.shape().to_vec()
    // }

    // fn dtype(&self) -> String {
    //     match self.inner.dtype() {
    //         DTYPE_FLOAT32 => "float32".to_string(),
    //         DTYPE_INT32 => "int32".to_string(),
    //         _ => "unknown".to_string(),
    //     }
    // }

    // fn device(&self) -> String {
    //     self.inner.device().to_string()
    // }

    // fn __add__(&self, other: &PyTensorView) -> Self {
    //     PyTensorView {
    //         inner: self.inner.clone() + other.inner.clone(),
    //     }
    // }

    // fn to_tensor(&self) -> PyResult<PyTensor> {
    //     // 创建一个连续的副本（因为视图可能非连续）
    //     let mut out = self
    //         .inner
    //         .create_output()
    //         .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    //     self.inner
    //         .contiguous(&mut out)
    //         .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    //     Ok(PyTensor {
    //         inner: out.into_handle(),
    //     })
    // }

    // fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
    //     let view = self
    //         .inner
    //         .broadcast_to(&shape)
    //         .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    //     Ok(PyTensorView { inner: view })
    // }

    // fn transpose(&self, axes: Vec<usize>) -> PyResult<Self> {
    //     let view = self
    //         .inner
    //         .transpose(&axes)
    //         .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    //     Ok(PyTensorView { inner: view })
    // }

    // fn T(&self) -> PyResult<Self> {
    //     let view = self.inner.T().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    //     Ok(PyTensorView { inner: view })
    // }
}
