//! TensorViewOps 核心 trait

use super::slice::SliceInfo;
use crate::device::Device;
use crate::dtype::DType;

pub trait TensorViewOps: Clone {
    type Handle: Clone;
    fn new(handle: Self::Handle) -> Self;
    fn as_strided(&self, new_shape: Vec<usize>, new_strides: Vec<usize>, offset: usize) -> Self;
    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String>;
    fn transpose(&self, axes: &[usize]) -> Result<Self, String>;
    fn T(&self) -> Result<Self, String> {
        let mut axes: Vec<usize> = (0..self.shape().len()).rev().collect();
        self.transpose(&axes)
    }
    fn is_contiguous(&self) -> bool;
    fn concat_with_out(views: &[&Self], axis: usize, out: &mut Self) -> Result<(), String>;
    fn split_with_outs(
        &self,
        sizes: &[usize],
        axis: usize,
        out_views: &mut [Self],
    ) -> Result<(), String>;
    fn concat(views: &[&Self], axis: usize) -> Result<Self, String>;
    fn split(&self, sizes: &[usize], axis: usize) -> Result<Vec<Self>, String>;
    fn strided_copy_to(&self, dst: &mut Self) -> Result<(), String>;
    fn contiguous(&self, out: &mut Self) -> Result<(), String>;
    /// 带输出张量的设备间传输（目标张量必须已分配）
    fn to(&self, out: &mut Self, target_device: Device) -> Result<(), String>;

    fn to_device(&self, target_device: Device) -> Result<Self, String>;

    fn to_cpu(&self) -> Result<Self, String> {
        self.to_device(Device::Cpu)
    }

    fn to_gpu(&self, device_id: usize) -> Result<Self, String> {
        self.to_device(Device::Cuda(device_id))
    }
    fn matmul_with_out(&self, other: &Self, out: &mut Self) -> Result<(), String>;
    fn matmul(&self, other: &Self) -> Result<Self, String>;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn dtype(&self) -> DType;
    fn size(&self) -> usize;
    fn assign(&mut self, src: &Self) -> Result<(), String>;
    fn slice(&self, info: &SliceInfo) -> Result<Self, String>;
}
