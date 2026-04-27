use crate::dtype::DTypeMapping;
use crate::view::TensorViewOps;
use crate::{ArcTensorView, RcTensorView};

macro_rules! impl_fill_for_view {
    (RcTensorView) => {
        impl RcTensorView {
            pub fn fill_cuda<T: bytemuck::Pod + DTypeMapping>(
                output: &mut Self,
                value: T,
            ) -> anyhow::Result<()> {
                use crate::cuda::get_stream;
                use crate::device::Device;
                use crate::tensor::Tensor;

                if !output.is_contiguous() {
                    anyhow::bail!("Output must be contiguous");
                }

                let scalar = Tensor::from_vec(vec![value], vec![1], output.device())?;
                let scalar_view = RcTensorView::new(scalar.into_rc());
                let broadcast_scalar = scalar_view.broadcast_to(output.shape())?;
                Self::elementwise_kernel(output, "out = in0", vec![&broadcast_scalar])
            }
        }
    };
    (ArcTensorView) => {
        impl ArcTensorView {
            pub fn fill_cuda<T: bytemuck::Pod + DTypeMapping>(
                output: &mut Self,
                value: T,
            ) -> anyhow::Result<()> {
                use crate::cuda::get_stream;
                use crate::device::Device;
                use crate::tensor::Tensor;

                if !output.is_contiguous() {
                    anyhow::bail!("Output must be contiguous");
                }

                let scalar = Tensor::from_vec(vec![value], vec![1], output.device())?;
                let scalar_view = ArcTensorView::new(scalar.into_arc());
                let broadcast_scalar = scalar_view.broadcast_to(output.shape())?;
                Self::elementwise_kernel(output, "out = in0", vec![&broadcast_scalar])
            }
        }
    };
}

// 为两种视图生成实现
impl_fill_for_view!(RcTensorView);
impl_fill_for_view!(ArcTensorView);
