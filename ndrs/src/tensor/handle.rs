/// Tensor 句柄类型定义（RcTensor, ArcTensor）及基础方法
use super::Tensor;
use crate::{DType, Device};
use anyhow::{Context, Result, anyhow, bail};
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// 引用计数（非线程安全）张量句柄。
#[derive(Clone, Debug)]
pub struct RcTensor(pub Rc<RefCell<Tensor>>);

impl RcTensor {
    /// 获取内部 `RefCell<Tensor>` 的只读引用（不获取锁）
    pub fn lock(&self) -> &RefCell<Tensor> {
        &*self.0
    }

    /// 从 `Tensor` 创建新的 `RcTensor`
    pub fn from_tensor(t: Tensor) -> Self {
        RcTensor(Rc::new(RefCell::new(t)))
    }

    /// 获取内部句柄的克隆
    pub fn into_inner(self) -> Rc<RefCell<Tensor>> {
        self.0
    }
}

impl From<Tensor> for RcTensor {
    fn from(t: Tensor) -> Self {
        RcTensor::from_tensor(t)
    }
}

/// 原子引用计数（线程安全）张量句柄。
#[derive(Clone, Debug)]
pub struct ArcTensor(pub Arc<ReentrantMutex<RefCell<Tensor>>>);

impl ArcTensor {
    /// 获取内部数据的互斥锁守卫
    pub fn lock(&self) -> parking_lot::ReentrantMutexGuard<RefCell<Tensor>> {
        self.0.lock()
    }

    /// 从 `Tensor` 创建新的 `ArcTensor`
    pub fn from_tensor(t: Tensor) -> Self {
        ArcTensor(Arc::new(ReentrantMutex::new(RefCell::new(t))))
    }

    /// 获取内部句柄的克隆
    pub fn into_inner(self) -> Arc<ReentrantMutex<RefCell<Tensor>>> {
        self.0
    }
}

impl From<Tensor> for ArcTensor {
    fn from(t: Tensor) -> Self {
        ArcTensor::from_tensor(t)
    }
}

impl ArcTensor {
    pub fn shape(&self) -> Vec<usize> {
        self.0.lock().borrow().shape().to_vec()
    }
    pub fn dtype(&self) -> DType {
        self.0.lock().borrow().dtype()
    }
    pub fn device(&self) -> Device {
        self.0.lock().borrow().device()
    }
}
