use crate::device::{CudaContextWrapper, Device};
use crate::dtype::{get_dtype_info, DType, DTypeMapping};
use bytemuck::{cast_slice, Pod};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) enum DataPtr {
    Cpu(Box<[u8]>),
    Gpu(CudaSlice<u8>),
}

impl DataPtr {
    pub(crate) fn as_ptr(&self, stream: Option<&Arc<CudaStream>>) -> *const u8 {
        match self {
            DataPtr::Cpu(b) => b.as_ptr(),
            DataPtr::Gpu(s) => {
                let stream_ref = stream.expect("Stream required for GPU pointer");
                let (ptr, _sync) = s.device_ptr(stream_ref);
                ptr as *const u8
            }
        }
    }
    pub(crate) fn as_mut_ptr(&mut self, stream: Option<&Arc<CudaStream>>) -> *mut u8 {
        match self {
            DataPtr::Cpu(b) => b.as_mut_ptr(),
            DataPtr::Gpu(s) => {
                let stream_ref = stream.expect("Stream required for GPU pointer");
                let (ptr, _sync) = s.device_ptr_mut(stream_ref);
                ptr as *mut u8
            }
        }
    }
    pub(crate) fn is_gpu(&self) -> bool {
        matches!(self, DataPtr::Gpu(_))
    }
}

/// 纯数据容器，不包含操作方法
pub struct Tensor {
    pub(crate) data: DataPtr,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) cuda_ctx: Option<Arc<CudaContextWrapper>>,
}

impl Tensor {
    // ---------- 构造函数 ----------
    pub fn new_cpu_from_slice<T: Pod + DTypeMapping>(data: &[T], shape: Vec<usize>) -> Self {
        let dtype = T::DTYPE;
        let elem_size = std::mem::size_of::<T>();
        let bytes = cast_slice(data).to_vec().into_boxed_slice();
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        Tensor {
            data: DataPtr::Cpu(bytes),
            shape,
            strides,
            dtype,
            device: Device::CPU,
            cuda_ctx: None,
        }
    }
    pub fn new_cpu_from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }
    pub fn new_cpu_from_i32(data: Vec<i32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }
    pub fn new_cpu_from_bytes(
        bytes: Box<[u8]>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, String> {
        let elem_size = get_dtype_info(dtype).ok_or("Unknown dtype")?.size;
        let expected_size = shape.iter().product::<usize>() * elem_size;
        if bytes.len() != expected_size {
            return Err(format!(
                "Byte size mismatch: expected {}, got {}",
                expected_size,
                bytes.len()
            ));
        }
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        Ok(Tensor {
            data: DataPtr::Cpu(bytes),
            shape,
            strides,
            dtype,
            device: Device::CPU,
            cuda_ctx: None,
        })
    }
    pub(crate) fn compute_row_major_strides(shape: &[usize], elem_size: usize) -> Vec<usize> {
        let mut strides = vec![elem_size; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // ---------- 内部访问器 ----------
    pub(crate) fn data_ptr(&self, stream: Option<&Arc<CudaStream>>) -> *const u8 {
        self.data.as_ptr(stream)
    }
    pub(crate) fn data_mut_ptr(&mut self, stream: Option<&Arc<CudaStream>>) -> *mut u8 {
        self.data.as_mut_ptr(stream)
    }
    pub(crate) fn cuda_ctx_ref(&self) -> Option<&Arc<CudaContextWrapper>> {
        self.cuda_ctx.as_ref()
    }

    // ---------- 公开获取器 ----------
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_contiguous(&self) -> bool {
        let elem_size = get_dtype_info(self.dtype).unwrap().size;
        let expected = Self::compute_row_major_strides(&self.shape, elem_size);
        self.strides == expected
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            DataPtr::Cpu(b) => Some(b),
            _ => None,
        }
    }

    // ---------- 所有权转换 ----------
    pub fn into_rc(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }
    pub fn into_arc(self) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DTYPE_FLOAT32;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.size(), 3);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert!(t.is_contiguous());
        assert_eq!(t.strides(), &[4]);
    }

    #[test]
    fn test_tensor_bytes() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let bytes = t.as_bytes().unwrap();
        assert_eq!(bytes.len(), 8);
        let values: Vec<f32> =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 2).to_vec() };
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_tensor_into_rc() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let rc = t.into_rc();
        assert_eq!(rc.borrow().shape(), &[2]);
    }

    #[test]
    fn test_tensor_into_arc() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let arc = t.into_arc();
        assert_eq!(arc.lock().unwrap().shape(), &[2]);
    }
}
