use crate::_Tensor;
use ndrs::tensor::ArcTensor;
use ndrs::ArcTensorView;
use ndrs::{DType, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "_TensorView")]
pub struct _TensorView {
    pub(crate) inner: ArcTensorView,
}

#[pymethods]
impl _TensorView {
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn dtype(&self) -> String {
        match self.inner.dtype() {
            DTYPE_FLOAT32 => "float32".to_string(),
            DTYPE_INT32 => "int32".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    fn __add__(&self, other: &_TensorView) -> Self {
        _TensorView {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn contiguous(&self) -> PyResult<_Tensor> {
        let handle = self
            .inner
            .contiguous()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_Tensor { inner: handle })
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        let view = self
            .inner
            .broadcast_to(&shape)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }

    fn transpose(&self, axes: Vec<usize>) -> PyResult<Self> {
        let view = self
            .inner
            .transpose(&axes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }

    fn T(&self) -> PyResult<Self> {
        let view = self
            .inner
            .T()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 tensor 类
    m.add_class::<_TensorView>()
}
