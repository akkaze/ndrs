pub mod conversion;
pub mod creation;
/// 为 Tensor 句柄提供运算符支持（通过新类型包装）
use super::Tensor;
use super::handle::{ArcTensor, RcTensor};
use crate::dtype::DTypeMapping;
use crate::view::{ArcTensorView, AsView, RcTensorView, TensorViewOps};
use anyhow::{Context, Result, anyhow, bail};
use std::ops::{Add, AddAssign};

/// 宏：为包装类型生成所有视图相关操作
macro_rules! impl_tensor_wrapper {
    (
        $wrapper:ident,
        $view:ty,
        $convert:expr
    ) => {
        impl $wrapper {
            /// 获取内部句柄的视图
            pub fn as_view(&self) -> $view {
                <$view>::new(self.clone())
            }

            pub fn broadcast_to(&self, target_shape: &[usize]) -> anyhow::Result<$view> {
                let view = self.as_view();
                view.broadcast_to(target_shape)
            }

            pub fn transpose(&self, axes: &[usize]) -> anyhow::Result<$view> {
                let view = self.as_view();
                view.transpose(axes)
            }

            pub fn T(&self) -> anyhow::Result<$view> {
                let view = self.as_view();
                view.T()
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
    RcTensorView,
    Tensor::into_rc_raw // 原始转换（用于需要原始句柄的场合，但当前未使用）
);

// 为 ArcTensor 生成实现
impl_tensor_wrapper!(ArcTensor, ArcTensorView, Tensor::into_arc_raw);

impl RcTensor {
    pub fn fill<T: bytemuck::Pod + DTypeMapping>(&mut self, value: T) -> anyhow::Result<()> {
        let mut view = self.as_view();
        view.fill(value)
    }
}

impl ArcTensor {
    pub fn fill<T: bytemuck::Pod + DTypeMapping>(&mut self, value: T) -> anyhow::Result<()> {
        let mut view = self.as_view();
        view.fill(value)
    }
}
