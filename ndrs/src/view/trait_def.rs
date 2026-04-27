/// TensorViewOps 核心 trait
use super::slice::SliceInfo;
use crate::device::Device;
use crate::dtype::{DType, get_dtype_info};
use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};

pub trait TensorViewOps: Clone {
    type Handle: Clone;
    fn new(handle: Self::Handle) -> Self;
    fn as_strided(&self, new_shape: Vec<usize>, new_strides: Vec<usize>, offset: usize) -> Self;
    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self>;
    fn transpose(&self, axes: &[usize]) -> Result<Self>;
    fn T(&self) -> Result<Self> {
        let mut axes: Vec<usize> = (0..self.shape().len()).rev().collect();
        self.transpose(&axes)
    }
    fn is_contiguous(&self) -> bool;
    fn concat_into(views: &[&Self], axis: usize, out: &mut Self) -> Result<()>;
    fn split_into(&self, sizes: &[usize], axis: usize, out_views: &mut [Self]) -> Result<()>;
    fn concat(views: &[&Self], axis: usize) -> Result<Self>;
    fn split(&self, sizes: &[usize], axis: usize) -> Result<Vec<Self>>;
    fn strided_copy_to(&self, dst: &mut Self) -> Result<()>;
    fn contiguous_into(&self, out: &mut Self) -> Result<()>;
    fn contiguous(&self) -> Result<Self::Handle>;
    fn to(&self, out: &mut Self, target_device: Device) -> Result<()>;
    fn to_device(&self, target_device: Device) -> Result<Self>;
    fn to_cpu(&self) -> Result<Self> {
        self.to_device(Device::Cpu)
    }
    fn to_gpu(&self, device_id: usize) -> Result<Self> {
        self.to_device(Device::Cuda(device_id))
    }
    fn matmul_into(&self, other: &Self, out: &mut Self) -> Result<()>;
    fn matmul(&self, other: &Self) -> Result<Self>;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn dtype(&self) -> DType;
    fn size(&self) -> usize;
    fn device(&self) -> Device;
    fn assign(&mut self, src: &Self) -> Result<()>;
    fn slice(&self, info: &SliceInfo) -> Result<Self>;
    fn num_bytes(&self) -> usize {
        let elem_size = get_dtype_info(self.dtype()).unwrap().size;
        self.size() * elem_size
    }
    fn fill<T: bytemuck::Pod + crate::dtype::DTypeMapping>(&mut self, value: T) -> Result<()>;

    unsafe fn raw_data_ptr(&self) -> *mut u8;

    #[cfg(feature = "cuda")]
    unsafe fn as_gpu_slice(&self) -> &CudaSlice<u8>;

    #[cfg(feature = "cuda")]
    unsafe fn as_gpu_slice_mut(&mut self) -> &mut CudaSlice<u8>;

    #[cfg(feature = "cuda")]
    unsafe fn as_gpu_view(&self) -> CudaView<'_, u8>;

    #[cfg(feature = "cuda")]
    unsafe fn as_gpu_view_mut(&self) -> CudaViewMut<'_, u8>;
}
pub trait AsView {
    type View: TensorViewOps;
    fn as_view(&self) -> Self::View;
}
