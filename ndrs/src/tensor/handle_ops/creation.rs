use crate::dtype::DTypeMapping;
use crate::tensor::{ArcTensor, RcTensor};
use crate::{DType, Device, Tensor};
use anyhow::{Context, Result, anyhow, bail};

// 在 handle.rs 或 handle_ops.rs 中添加
macro_rules! impl_tensor_handle_constructors {
    ($handle_type:ident, $into_method:ident) => {
        impl $handle_type {
            /// 创建指定形状和数据类型的零张量
            pub fn zeros(shape: Vec<usize>, dtype: DType, device: Device) -> anyhow::Result<Self> {
                let tensor = Tensor::new_contiguous(shape, dtype, device)?;
                Ok(tensor.$into_method())
            }

            /// 创建指定形状和数据类型的全一张量
            pub fn ones(shape: Vec<usize>, dtype: DType, device: Device) -> anyhow::Result<Self> {
                let tensor = Tensor::new_contiguous(shape, dtype, device)?;
                let mut handle = tensor.$into_method();
                match dtype {
                    crate::dtype::DTYPE_FLOAT32 => handle.fill(1.0f32)?,
                    crate::dtype::DTYPE_INT32 => handle.fill(1i32)?,
                    _ => anyhow::bail!("Unsupported dtype for ones: {}", dtype),
                }
                Ok(handle)
            }

            /// 创建指定形状和数据类型的空张量（未初始化，当前为零初始化）
            pub fn empty(shape: Vec<usize>, dtype: DType, device: Device) -> anyhow::Result<Self> {
                // 新连续张量默认零初始化，所以 empty 和 zeros 目前无区别
                Self::zeros(shape, dtype, device)
            }
        }
    };
}

impl_tensor_handle_constructors!(RcTensor, into_rc);
impl_tensor_handle_constructors!(ArcTensor, into_arc);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};
    use crate::{TensorViewOps, cuda};

    #[test]
    fn test_zeros() {
        let t = RcTensor::zeros(vec![2, 2], DTYPE_FLOAT32, Device::Cpu).unwrap();
        let view = t.as_view();
        let data = view.contiguous().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones_f32() {
        let t = RcTensor::ones(vec![2, 3], DTYPE_FLOAT32, Device::Cpu).unwrap();
        let view = t.as_view();
        let data = view.contiguous().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_ones_gpu_i32() {
        if !cuda::is_available() {
            return;
        }
        let t = ArcTensor::ones(vec![2, 2, 2, 2], DTYPE_INT32, Device::Cuda(0)).unwrap();
        let view = t.as_view();
        let data = view
            .to_cpu()
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec::<i32>()
            .unwrap();
        assert_eq!(data, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_empty() {
        let t = RcTensor::empty(vec![2], DTYPE_INT32, Device::Cpu).unwrap();
        let view = t.as_view();
        let data = view.contiguous().unwrap().to_vec::<i32>().unwrap();
        assert_eq!(data, vec![0, 0]); // 零初始化
    }
}
