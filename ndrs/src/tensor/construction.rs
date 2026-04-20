use super::*;
use crate::dtype::{get_dtype_info, DType, DTypeMapping};
use crate::device::Device;
use crate::tensor::{DataPtr, Tensor};
use bytemuck::{cast_slice, Pod};
use cudarc::driver::DevicePtr;

impl Tensor {
    // ---------- 构造函数 ----------
    pub fn new_cpu_from_slice<T: Pod + DTypeMapping>(data: &[T], shape: Vec<usize>) -> Self {
        let dtype = T::DTYPE;
        let elem_size = std::mem::size_of::<T>();
        let bytes = cast_slice(data).to_vec().into_boxed_slice();
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        Tensor {
            data: DataPtr::Cpu(bytes),
            shape,
            strides,
            dtype,
            device: Device::CPU,
            cuda_ctx: None,
        }
    }
    pub fn new_cpu_from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }
    pub fn new_cpu_from_i32(data: Vec<i32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }
    pub fn new_cpu_from_bytes(
        bytes: Box<[u8]>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, String> {
        let elem_size = get_dtype_info(dtype).ok_or("Unknown dtype")?.size;
        let expected_size = shape.iter().product::<usize>() * elem_size;
        if bytes.len() != expected_size {
            return Err(format!(
                "Byte size mismatch: expected {}, got {}",
                expected_size,
                bytes.len()
            ));
        }
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        Ok(Tensor {
            data: DataPtr::Cpu(bytes),
            shape,
            strides,
            dtype,
            device: Device::CPU,
            cuda_ctx: None,
        })
    }
    /// 创建一个形状为 `shape`、数据类型为 `dtype` 的连续 CPU 张量，内存初始化为零。
    pub fn new_contiguous(shape: Vec<usize>, dtype: DType) -> Result<Self, String> {
        let elem_size = get_dtype_info(dtype).ok_or("Unknown dtype")?.size;
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * elem_size;
        let bytes = vec![0u8; total_bytes].into_boxed_slice();
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        Ok(Tensor {
            data: DataPtr::Cpu(bytes),
            shape,
            strides,
            dtype,
            device: Device::CPU,
            cuda_ctx: None,
        })
    }
}