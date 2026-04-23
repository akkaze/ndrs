//! 为 Tensor 句柄提供运算符支持（通过新类型包装）

use super::*;
use crate::view::{ArcTensorView, AsView, RcTensorView, TensorViewOps};
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::ops::{Add, AddAssign};

/// 宏：为包装类型生成所有实现
macro_rules! impl_tensor_wrapper {
    (
        $wrapper:ident,
        $handle:ty,
        $view:ty,
        $convert:expr
    ) => {
        #[derive(Clone, Debug)]
        pub struct $wrapper(pub $handle);

        impl From<Tensor> for $wrapper {
            fn from(t: Tensor) -> Self {
                $wrapper($convert(t))
            }
        }

        impl $wrapper {
            /// 获取内部句柄的视图
            pub fn as_view(&self) -> $view {
                <$view>::new(self.clone())
            }

            /// 获取底层句柄的克隆
            pub fn into_inner(self) -> $handle {
                self.0
            }

            pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
                let view = self.as_view();
                let result_view = view.broadcast_to(target_shape)?;
                Ok(result_view.into_handle()) // 直接返回，因为 into_handle 返回 Self
            }

            pub fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
                let view = self.as_view();
                let result_view = view.transpose(axes)?;
                Ok(result_view.into_handle())
            }

            pub fn T(&self) -> Result<Self, String> {
                let view = self.as_view();
                let result_view = view.T()?;
                Ok(result_view.into_handle())
            }
        }

        impl Add for $wrapper {
            type Output = Self;
            fn add(self, other: Self) -> Self::Output {
                let a_view = self.as_view();
                let b_view = other.as_view();
                let result_view = a_view + b_view;
                result_view.into_handle()
            }
        }

        impl AddAssign for $wrapper {
            fn add_assign(&mut self, other: Self) {
                let mut a_view = self.as_view();
                let b_view = other.as_view();
                a_view += b_view;
                // a_view 持有 self 的引用，底层数据已修改
            }
        }

        impl AsView for $wrapper {
            type View = $view;
            fn as_view(&self) -> Self::View {
                <$view>::new(self.clone())
            }
        }
    };
}

// 为 RcTensor 生成实现
impl_tensor_wrapper!(
    RcTensor,
    std::rc::Rc<std::cell::RefCell<Tensor>>,
    RcTensorView,
    Tensor::into_rc_raw // 注意：原始转换
);

// 为 ArcTensor 生成实现
impl_tensor_wrapper!(
    ArcTensor,
    std::sync::Arc<ReentrantMutex<RefCell<Tensor>>>,
    ArcTensorView,
    Tensor::into_arc_raw // 原始转换
);
