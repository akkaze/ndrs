use crate::tensor::PyTensor;
use crate::utils::flatten_list_f32;
use ndrs::{ArcTensorView, Device, Tensor, TensorViewOps, DTYPE_FLOAT32, DTYPE_INT32};
use numpy::{PyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn tensor_new_impl(data: &PyAny, dtype: Option<u32>) -> PyResult<PyTensor> {
    let target_dtype = if let Some(d) = dtype {
        d
    } else {
        let is_int = if let Ok(list) = data.extract::<&PyList>() {
            list.iter().all(|item| item.extract::<i64>().is_ok())
        } else if let Ok(_arr) = data.extract::<&PyArrayDyn<i32>>() {
            true
        } else {
            false
        };
        if is_int {
            DTYPE_INT32
        } else {
            DTYPE_FLOAT32
        }
    };
    match target_dtype {
        DTYPE_FLOAT32 => {
            let (data_f32, shape) = if let Ok(arr) = data.extract::<&PyArrayDyn<f32>>() {
                (
                    arr.readonly().as_slice().unwrap().to_vec(),
                    arr.shape().iter().map(|&s| s as usize).collect(),
                )
            } else {
                let mut shape = Vec::new();
                let data = flatten_list_f32(data, &mut shape)?;
                (data, shape)
            };
            let tensor = Tensor::new_cpu_from_f32(data_f32, shape);
            let view = ArcTensorView::new(tensor.into_arc());
            Ok(PyTensor::from_view(view))
        }
        DTYPE_INT32 => {
            let data_i32: Vec<i32> = if let Ok(arr) = data.extract::<&PyArrayDyn<i32>>() {
                arr.readonly().as_slice().unwrap().to_vec()
            } else if let Ok(list) = data.extract::<&PyList>() {
                list.extract::<Vec<i64>>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Cannot convert data to i32",
                ));
            };
            let shape = vec![data_i32.len()];
            let tensor = Tensor::new_cpu_from_i32(data_i32, shape);
            let view = ArcTensorView::new(tensor.into_arc());
            Ok(PyTensor::from_view(view))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err("Unsupported dtype")),
    }
}

pub fn tensor_from_numpy_impl(arr: &PyAny) -> PyResult<PyTensor> {
    if let Ok(arr_f32) = arr.extract::<&PyArrayDyn<f32>>() {
        let shape = arr_f32.shape().iter().map(|&s| s as usize).collect();
        let vec = arr_f32.readonly().as_slice().unwrap().to_vec();
        let tensor = Tensor::new_cpu_from_f32(vec, shape);
        let view = ArcTensorView::new(tensor.into_arc());
        return Ok(PyTensor::from_view(view));
    }
    if let Ok(arr_i32) = arr.extract::<&PyArrayDyn<i32>>() {
        let shape = arr_i32.shape().iter().map(|&s| s as usize).collect();
        let vec = arr_i32.readonly().as_slice().unwrap().to_vec();
        let tensor = Tensor::new_cpu_from_i32(vec, shape);
        let view = ArcTensorView::new(tensor.into_arc());
        return Ok(PyTensor::from_view(view));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported numpy array dtype, only float32 and int32 are supported",
    ))
}

pub fn tensor_numpy_impl(view: &ArcTensorView, py: Python) -> PyResult<Py<PyAny>> {
    let elem_size = ndrs::dtype::get_dtype_info(view.dtype()).unwrap().size;
    let total_bytes = view.shape().iter().product::<usize>() * elem_size;
    let out_tensor = Tensor::new_cpu_from_bytes(
        vec![0u8; total_bytes].into_boxed_slice(),
        view.shape().to_vec(),
        view.dtype(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let out_handle = out_tensor.into_arc();
    let mut out_view = ArcTensorView::new(out_handle);
    view.contiguous(&mut out_view)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let tensor_contig = out_view.into_handle();
    let tensor_contig_ref = tensor_contig.0.lock().unwrap();
    let bytes = tensor_contig_ref
        .as_bytes()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Cannot get bytes"))?;
    let shape = tensor_contig_ref.shape();

    match tensor_contig_ref.dtype() {
        DTYPE_FLOAT32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, tensor_contig_ref.size())
            };
            let array = PyArray::from_vec(py, slice.to_vec())
                .reshape(shape)
                .map_err(|e: PyErr| e)?;
            Ok(array.into())
        }
        DTYPE_INT32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const i32, tensor_contig_ref.size())
            };
            let array = PyArray::from_vec(py, slice.to_vec())
                .reshape(shape)
                .map_err(|e: PyErr| e)?;
            Ok(array.into())
        }
        _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Unsupported dtype for numpy conversion",
        )),
    }
}

pub fn tensor_to_impl(view: &ArcTensorView, device: &str) -> PyResult<PyTensor> {
    let target = match device {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown device")),
    };
    let elem_size = ndrs::dtype::get_dtype_info(view.dtype()).unwrap().size;
    let total_bytes = view.shape().iter().product::<usize>() * elem_size;
    let out_tensor = Tensor::new_cpu_from_bytes(
        vec![0u8; total_bytes].into_boxed_slice(),
        view.shape().to_vec(),
        view.dtype(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let out_handle = out_tensor.into_arc();
    let mut out_view = ArcTensorView::new(out_handle);
    view.to(&mut out_view, target)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(PyTensor::from_view(out_view))
}
