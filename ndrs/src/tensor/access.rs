use super::*;
use crate::dtype::{DType, get_dtype_info};
use crate::tensor::Tensor;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

impl Tensor {
    // ---------- 内部访问器 ----------
    pub(crate) fn data_ptr(&self, stream: Option<&Arc<CudaStream>>) -> *const u8 {
        self.data.as_ptr(stream)
    }
    pub(crate) fn data_mut_ptr(&mut self, stream: Option<&Arc<CudaStream>>) -> *mut u8 {
        self.data.as_mut_ptr(stream)
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

    // ---------- 辅助函数 ----------
    pub(crate) fn compute_row_major_strides(shape: &[usize], elem_size: usize) -> Vec<usize> {
        let mut strides = vec![elem_size; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn as_cpu_slice(&self) -> Option<&[u8]> {
        match &self.data {
            DataPtr::Cpu(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_cpu_slice_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.data {
            DataPtr::Cpu(b) => Some(b),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_gpu_slice(&self) -> Option<&CudaSlice<u8>> {
        match &self.data {
            DataPtr::Gpu(s) => Some(s),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_gpu_slice_mut(&mut self) -> Option<&mut CudaSlice<u8>> {
        match &mut self.data {
            DataPtr::Gpu(s) => Some(s),
            _ => None,
        }
    }
}
