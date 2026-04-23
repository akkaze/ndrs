use ndrs::{ArcTensorView, Device, Tensor, TensorViewOps};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass(name = "Tensor")]
pub struct PyTensor {
    inner: ArcTensorView,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn from_list(list: &Bound<'_, PyList>, device: Option<String>) -> PyResult<Self> {
        let shape = vec![list.len()];
        let mut data = Vec::with_capacity(list.len());
        for elem in list.iter() {
            let f = elem.extract::<f64>()? as f32;
            data.push(f);
        }
        let tensor = Tensor::new_cpu_from_f32(data, shape);
        let view = tensor.into_arc().as_view();
        let device = device.unwrap_or_else(|| "cpu".to_string());
        let dev = parse_device(&device).map_err(|e| PyRuntimeError::new_err(e))?;
        let view = if dev == Device::Cpu {
            view.to_cpu().map_err(|e| PyRuntimeError::new_err(e))?
        } else {
            let id = dev.as_cuda_index().unwrap();
            view.to_gpu(id).map_err(|e| PyRuntimeError::new_err(e))?
        };
        Ok(PyTensor { inner: view })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn dtype(&self) -> String {
        match self.inner.dtype() {
            ndrs::DTYPE_FLOAT32 => "float32".to_string(),
            ndrs::DTYPE_INT32 => "int32".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn device(&self) -> String {
        let guard = self.inner.handle().0.lock();
        let tensor = guard.borrow();
        tensor.device().to_string()
    }

    fn to_cpu(&self) -> PyResult<Self> {
        let cpu_view = self
            .inner
            .to_cpu()
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(PyTensor { inner: cpu_view })
    }

    fn to_gpu(&self, device_id: usize) -> PyResult<Self> {
        let gpu_view = self
            .inner
            .to_gpu(device_id)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(PyTensor { inner: gpu_view })
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.clone() + other.inner.clone();
        Ok(PyTensor { inner: result })
    }

    fn numpy(&self) -> PyResult<Vec<f32>> {
        let cpu_view = self
            .inner
            .to_cpu()
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let guard = cpu_view.handle().0.lock();
        let tensor = guard.borrow();
        let bytes = tensor
            .as_bytes()
            .ok_or_else(|| PyRuntimeError::new_err("No bytes"))?;
        let slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.inner.size()) };
        Ok(slice.to_vec())
    }
}

fn parse_device(s: &str) -> Result<Device, String> {
    s.parse()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
