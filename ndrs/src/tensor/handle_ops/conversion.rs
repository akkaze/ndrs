use crate::dtype::DTypeMapping;
use crate::tensor::{ArcTensor, RcTensor};
use crate::{Device, Tensor};
use bytemuck::Pod;
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

macro_rules! impl_tensor_handle_conversions {
    ($handle_type:ident) => {
        impl $handle_type {
            pub fn to_vec<T: Pod + DTypeMapping>(&self) -> anyhow::Result<Vec<T>> {
                let guard = self.lock();
                let tensor = guard.borrow();
                tensor.to_vec()
            }

            pub fn from_vec<T: bytemuck::Pod + DTypeMapping>(
                data: Vec<T>,
                shape: Vec<usize>,
                device: Device,
            ) -> anyhow::Result<Self> {
                let tensor = Tensor::from_vec(data, shape, device)?;
                Ok(Self::from_tensor(tensor))
            }

            pub fn from_scalar<T: bytemuck::Pod + DTypeMapping>(
                value: T,
                device: Device,
            ) -> anyhow::Result<Self> {
                let tensor = Tensor::from_scalar(value, device)?;
                Ok(Self::from_tensor(tensor))
            }
        }
    };
}

impl_tensor_handle_conversions!(RcTensor);
impl_tensor_handle_conversions!(ArcTensor);
