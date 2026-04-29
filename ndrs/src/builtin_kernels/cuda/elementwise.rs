use crate::Device;
use crate::dtype::DTypeMapping;
use crate::view::TensorViewOps;
use crate::{ArcTensorView, RcTensorView};
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[macro_export]
macro_rules! impl_elementwise_kernel_for_view {
    ($view_type:ident) => {
        impl $view_type {
            pub fn elementwise_kernel(
                output: &mut Self,
                expression: &str,
                inputs: Vec<&Self>,
            ) -> anyhow::Result<()> {
                use crate::backend::cuda::ElementwiseKernel;
                use crate::cuda::get_stream;
                use anyhow::Context;

                let stream = get_stream().context("Failed to get CUDA stream")?;
                let ctx = stream.inner().context().clone();

                let shape = output.shape();
                let device = output.device();
                if device == Device::Cpu {
                    anyhow::bail!("Kernel only works on GPU tensors");
                }
                for inp in &inputs {
                    if inp.device() != device {
                        anyhow::bail!(
                            "All views must be on same device: expected {:?}, got {:?}",
                            device,
                            inp.device()
                        );
                    }
                }

                // 构造参数列表字符串
                let mut params_str = String::from("V out");
                for i in 0..inputs.len() {
                    params_str.push_str(&format!(", V in{}", i));
                }
                let expr = if expression.contains('=') {
                    expression.to_string()
                } else {
                    format!("out = {}", expression)
                };

                let dev_id = match device {
                    Device::Cuda(id) => id,
                    _ => 0,
                };
                let kernel_name = format!("elementwise_{}_{}", shape.len(), dev_id);
                let mut kernel =
                    ElementwiseKernel::from_expression(&params_str, &expr, &kernel_name, ctx)?;

                kernel.launch_views(output, inputs, None)
            }
        }
    };
}

impl_elementwise_kernel_for_view!(RcTensorView);
impl_elementwise_kernel_for_view!(ArcTensorView);
