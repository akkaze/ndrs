use crate::cuda::stream::PyCudaStream;
use crate::view::_TensorView;
use cudarc::driver::PushKernelArg;
use ndrs::backend::cuda::ElementwiseKernel;
use ndrs::cuda::get_stream;
use ndrs::view::TensorViewOps;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "_ElementwiseKernel")]
pub struct PyElementwiseKernel {
    inner: ElementwiseKernel,
}

#[pymethods]
impl PyElementwiseKernel {
    #[new]
    fn new(param_str: &str, expr: &str, name: &str) -> PyResult<Self> {
        let stream = get_stream().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ctx = stream.inner().context().clone();
        let kernel = ElementwiseKernel::from_expression(param_str, expr, name, ctx)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyElementwiseKernel { inner: kernel })
    }

    fn launch(
        &mut self,
        py: Python<'_>,
        inputs: Vec<PyRef<_TensorView>>,
        mut output: PyRefMut<_TensorView>,
        stream: Option<&PyCudaStream>,
    ) -> PyResult<()> {
        let stream = match stream {
            Some(s) => s.inner.clone(),
            None => get_stream().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        ndrs::cuda::set_stream(stream.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let input_views: Vec<_> = inputs.iter().map(|v| &v.inner).collect();
        self.inner
            .launch_views(&mut output.inner, input_views, None)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElementwiseKernel>()?;
    Ok(())
}
