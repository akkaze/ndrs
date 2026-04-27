use super::*;
use crate::dtype::{DTypeMapping, get_dtype_info};
use crate::tensor::{DataPtr, Tensor};
use anyhow::{Context, Result, anyhow, bail};
use bytemuck::Pod;
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

impl Tensor {
    /// 将张量数据转换为 `Vec<T>`（要求张量是连续的且数据类型匹配）
    pub fn to_vec<T: Pod + DTypeMapping>(&self) -> anyhow::Result<Vec<T>> {
        if self.dtype() != T::DTYPE {
            bail!("Type mismatch");
        }
        if !self.is_contiguous() {
            bail!("Tensor must be contiguous");
        }
        let bytes = self.as_bytes().context("Cannot get bytes")?;
        let elem_size = std::mem::size_of::<T>();
        if bytes.len() % elem_size != 0 {
            bail!("Byte size not multiple of element size");
        }
        let n = bytes.len() / elem_size;
        let slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) };
        Ok(slice.to_vec())
    }

    /// 从 `Vec<T>` 创建连续 CPU 张量
    pub fn from_vec<T: Pod + DTypeMapping>(
        data: Vec<T>,
        shape: Vec<usize>,
        device: Device,
    ) -> anyhow::Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            bail!("Data length does not match shape");
        }
        let bytes = bytemuck::cast_slice(&data).to_vec().into_boxed_slice();
        Self::new_from_bytes(bytes, shape, T::DTYPE, device)
    }

    pub fn from_scalar<T: Pod + DTypeMapping>(value: T, device: Device) -> anyhow::Result<Self> {
        Self::from_vec(vec![value], vec![], device)
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            DataPtr::Cpu(b) => Some(b),
            _ => None,
        }
    }

    // ---------- 所有权转换 ----------
    pub fn into_rc(self) -> RcTensor {
        RcTensor(Rc::new(RefCell::new(self)))
    }
    pub fn into_arc(self) -> ArcTensor {
        ArcTensor(Arc::new(ReentrantMutex::new(RefCell::new(self))))
    }

    // 内部使用：返回原始句柄
    pub(crate) fn into_rc_raw(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }

    pub(crate) fn into_arc_raw(self) -> Arc<ReentrantMutex<RefCell<Tensor>>> {
        Arc::new(ReentrantMutex::new(RefCell::new(self)))
    }
}
