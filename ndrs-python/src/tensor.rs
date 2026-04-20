use ndrs::{ArcTensorView, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use pyo3::prelude::*;

// 导入辅助函数
use crate::conversion::{tensor_new_impl, tensor_from_numpy_impl, tensor_numpy_impl, tensor_to_impl};
use crate::ops::tensor_add_impl;

// ---------- PyTensor ----------
#[pyclass(name = "Tensor", unsendable)]
pub struct PyTensor {
    pub inner: ArcTensorView,
}

impl PyTensor {
    pub fn from_view(view: ArcTensorView) -> Self {
        PyTensor { inner: view }
    }
}

#[pymethods]
impl PyTensor {
    // 基础属性
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> u32 {
        self.inner.dtype()
    }

    #[getter]
    fn device(&self) -> crate::device::PyDevice {
        crate::device::PyDevice {
            inner: self.inner.handle().lock().unwrap().device(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={:?})",
            self.inner.shape(),
            self.inner.dtype(),
            self.inner.handle().lock().unwrap().device()
        )
    }

    // 视图
    fn view(&self) -> PyResult<PyTensorView> {
        Ok(PyTensorView { inner: self.inner.clone() })
    }

    // 构造方法
    #[new]
    #[pyo3(signature = (data, dtype=None, _device=None))]
    fn new(data: &PyAny, dtype: Option<u32>, _device: Option<crate::device::PyDevice>) -> PyResult<Self> {
        tensor_new_impl(data, dtype)
    }

    #[staticmethod]
    fn from_numpy(arr: &PyAny, _dtype: Option<u32>, _device: Option<crate::device::PyDevice>) -> PyResult<Self> {
        tensor_from_numpy_impl(arr)
    }

    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        tensor_numpy_impl(&self.inner, py)
    }

    fn to(&self, device: &str) -> PyResult<Self> {
        tensor_to_impl(&self.inner, device)
    }

    // 运算符
    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        tensor_add_impl(&self.inner, &other.inner)
    }

    // 可以继续添加 __sub__, __mul__, __matmul__, __iadd__ 等
}

// ---------- PyTensorView ----------
#[pyclass(name = "TensorView", unsendable)]
pub struct PyTensorView {
    pub inner: ArcTensorView,
}

#[pymethods]
impl PyTensorView {
    fn as_strided(&self, new_shape: Vec<usize>, new_strides: Vec<usize>, offset: usize) -> Self {
        PyTensorView {
            inner: self.inner.as_strided(new_shape, new_strides, offset),
        }
    }

    fn broadcast_to(&self, target_shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self.inner.broadcast_to(&target_shape)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }

    fn transpose(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self.inner.transpose(&axes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }

    fn T(&self) -> PyResult<Self> {
        Ok(PyTensorView {
            inner: self.inner.T()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        })
    }

    fn contiguous(&mut self, out: &mut PyTensorView) -> PyResult<()> {
        self.inner.contiguous(&mut out.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    fn assign(&mut self, src: &PyTensorView) -> PyResult<()> {
        self.inner.assign(&src.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}