// ndrs/src/backend/cpu/fill.rs
use super::elementwise::*;
use crate::device::Device;
use crate::dtype::DTypeMapping;
use crate::tensor::Tensor;
use crate::view::TensorViewOps;
use crate::view::{ArcTensorView, AsView, RcTensorView};
use anyhow::Result;

impl RcTensorView {
    /// 填充 CPU 视图为常量值
    pub fn fill_cpu<T: bytemuck::Pod + DTypeMapping>(&mut self, value: T) -> Result<()> {
        let scalar = Tensor::from_vec(vec![value], vec![1], Device::Cpu)?;
        let scalar_view = RcTensorView::new(scalar.into_rc());
        let broadcast = scalar_view.broadcast_to(self.shape())?;
        RcTensorView::elementwise::<T, _>(self, &[&broadcast], |_| value)
    }
}

impl ArcTensorView {
    /// 填充 CPU 视图为常量值
    pub fn fill_cpu<T: bytemuck::Pod + DTypeMapping>(&mut self, value: T) -> Result<()> {
        let scalar = Tensor::from_vec(vec![value], vec![1], Device::Cpu)?;
        let scalar_view = ArcTensorView::new(scalar.into_arc());
        let broadcast = scalar_view.broadcast_to(self.shape())?;
        ArcTensorView::elementwise::<T, _>(self, &[&broadcast], |_| value)
    }
}
