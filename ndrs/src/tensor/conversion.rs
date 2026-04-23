use super::*;
use crate::dtype::{get_dtype_info, DTypeMapping};
use crate::tensor::{DataPtr, Tensor};
use bytemuck::Pod;
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

impl Tensor {
    /// 将张量数据转换为 `Vec<T>`（要求张量是连续的且数据类型匹配）
    pub fn to_vec<T: Pod + DTypeMapping>(&self) -> Result<Vec<T>, String> {
        if self.dtype() != T::DTYPE {
            return Err("Type mismatch".into());
        }
        if !self.is_contiguous() {
            return Err("Tensor must be contiguous".into());
        }
        let bytes = self.as_bytes().ok_or("Cannot get bytes")?;
        let elem_size = std::mem::size_of::<T>();
        if bytes.len() % elem_size != 0 {
            return Err("Byte size not multiple of element size".into());
        }
        let n = bytes.len() / elem_size;
        let slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) };
        Ok(slice.to_vec())
    }

    /// 从 `Vec<T>` 创建连续 CPU 张量
    pub fn from_vec<T: Pod + DTypeMapping>(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err("Data length does not match shape".into());
        }
        Ok(Self::new_cpu_from_slice(&data, shape))
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
