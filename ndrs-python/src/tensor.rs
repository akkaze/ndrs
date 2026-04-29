use crate::_TensorView;
use anyhow::{anyhow, bail, Context, Result};
use ndrs::tensor::ArcTensor;
use ndrs::{DType, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::str::FromStr;

fn parse_device(s: &str) -> anyhow::Result<Device> {
    s.parse()
}

#[pyclass(name = "_Tensor")]
pub struct _Tensor {
    pub(crate) inner: ArcTensor,
}

#[pymethods]
impl _Tensor {
    #[staticmethod]
    fn from_bytes(
        py: Python,
        bytes: &Bound<'_, PyBytes>,
        shape: Vec<usize>,
        dtype_id: u32, // 改为 u32
        device: Option<String>,
    ) -> PyResult<Self> {
        let dev = device.unwrap_or_else(|| "cpu".to_string());
        let dev = parse_device(&dev).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tensor = Tensor::new_from_bytes(
            bytes.as_bytes().to_vec().into_boxed_slice(),
            shape,
            dtype_id,
            dev,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_Tensor {
            inner: tensor.into_arc(),
        })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
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

    fn __add__(&self, other: &_Tensor) -> Self {
        _Tensor {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn as_bytes<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyBytes>> {
        let cpu_view = self
            .inner
            .as_view()
            .to_cpu()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let guard = cpu_view.handle().0.lock();
        let tensor = guard.borrow();
        let bytes = tensor
            .as_bytes()
            .ok_or_else(|| PyRuntimeError::new_err("No bytes"))?;
        Ok(PyBytes::new(py, bytes))
    }

    fn dtype_id(&self) -> u32 {
        self.inner.dtype()
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<_TensorView> {
        let view = self
            .inner
            .as_view()
            .broadcast_to(&shape)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }

    fn transpose(&self, axes: Vec<usize>) -> PyResult<_TensorView> {
        let view = self
            .inner
            .as_view()
            .transpose(&axes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }

    fn T(&self) -> PyResult<_TensorView> {
        let view = self
            .inner
            .as_view()
            .T()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_TensorView { inner: view })
    }

    fn as_view(&self) -> PyResult<_TensorView> {
        let view = self.inner.as_view();
        Ok(_TensorView { inner: view })
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>, dtype: u32, device: &str) -> PyResult<Self> {
        let device =
            Device::from_str(device).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tensor = ArcTensor::zeros(shape, dtype, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_Tensor { inner: tensor })
    }

    /// 创建全一张量
    #[staticmethod]
    fn ones(shape: Vec<usize>, dtype: u32, device: &str) -> PyResult<Self> {
        let device =
            Device::from_str(device).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tensor = ArcTensor::ones(shape, dtype, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_Tensor { inner: tensor })
    }

    /// 创建未初始化张量（当前为零）
    #[staticmethod]
    fn empty(shape: Vec<usize>, dtype: u32, device: &str) -> PyResult<Self> {
        let device =
            Device::from_str(device).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tensor = ArcTensor::empty(shape, dtype, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(_Tensor { inner: tensor })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 tensor 类
    m.add_class::<_Tensor>()
}
