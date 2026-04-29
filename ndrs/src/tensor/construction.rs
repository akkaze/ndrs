use super::*;
use crate::device::Device;
use crate::dtype::{DType, DTypeMapping, get_dtype_info};
use crate::tensor::{DataPtr, Tensor};
use crate::{DTYPE_FLOAT32, DTYPE_INT32};
use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, cast_slice};
use cudarc::driver::DevicePtr;

impl Tensor {
    #[cfg(feature = "cuda")]
    fn init_device_strides(
        &self,
    ) -> anyhow::Result<(Option<CudaSlice<usize>>, Option<CudaSlice<usize>>)> {
        match self.device {
            Device::Cpu => Ok((None, None)),
            Device::Cuda(dev_id) => {
                let stream = crate::cuda::get_stream()?;
                if stream.device_id != dev_id {
                    anyhow::bail!("Stream device mismatch");
                }
                let shape_dev = Some(stream.inner().clone_htod(self.shape())?);
                let strides_dev = Some(stream.inner().clone_htod(self.strides())?);
                Ok((shape_dev, strides_dev))
            }
        }
    }
}

impl Tensor {
    // ---------- 构造函数 ----------
    pub fn from_slice<T: Pod + DTypeMapping>(
        data: &[T],
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self> {
        let dtype = T::DTYPE;
        let elem_size = std::mem::size_of::<T>();
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            bail!(
                "Data length {} does not match shape product {}",
                data.len(),
                expected_size
            );
        }
        let bytes = cast_slice(data).to_vec().into_boxed_slice();
        Self::new_from_bytes(bytes, shape, dtype, device)
    }

    // ---------- 保留原有 CPU 专用方法（向后兼容）----------
    pub fn new_cpu_from_slice<T: Pod + DTypeMapping>(data: &[T], shape: Vec<usize>) -> Self {
        Self::from_slice(data, shape, Device::Cpu).expect("Failed to create CPU tensor")
    }

    pub fn new_cpu_from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }

    pub fn new_cpu_from_i32(data: Vec<i32>, shape: Vec<usize>) -> Self {
        Self::new_cpu_from_slice(&data, shape)
    }

    /// 通用字节构造器（支持 CPU / GPU）
    pub fn new_from_bytes(
        bytes: Box<[u8]>,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let elem_size = get_dtype_info(dtype).context("Unknown dtype")?.size;
        let expected_size = shape.iter().product::<usize>() * elem_size;
        if bytes.len() != expected_size {
            bail!(
                "Byte size mismatch: expected {}, got {}",
                expected_size,
                bytes.len()
            );
        }
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        let data = match device {
            Device::Cpu => DataPtr::Cpu(bytes),
            Device::Cuda(dev_id) => {
                use crate::cuda;
                let stream = cuda::get_stream()?;
                if stream.device_id != dev_id {
                    bail!(
                        "Stream device {} does not match target device {}",
                        stream.device_id,
                        dev_id
                    );
                }
                let mut gpu_mem = stream.inner().alloc_zeros::<u8>(bytes.len())?;
                stream.inner().memcpy_htod(bytes.as_ref(), &mut gpu_mem)?;
                DataPtr::Gpu(gpu_mem)
            }
        };
        Ok(Tensor {
            data,
            shape,
            strides,
            dtype,
            device,
        })
    }

    /// 从字节创建 CPU 张量（向后兼容）
    pub fn new_cpu_from_bytes(bytes: Box<[u8]>, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        Self::new_from_bytes(bytes, shape, dtype, Device::Cpu)
    }

    /// 创建一个形状为 `shape`、数据类型为 `dtype` 的连续张量，内存初始化为零。
    pub fn new_contiguous(shape: Vec<usize>, dtype: DType, device: Device) -> Result<Self> {
        let elem_size = get_dtype_info(dtype).context("Unknown dtype")?.size;
        let total_bytes = shape.iter().product::<usize>() * elem_size;
        let strides = Self::compute_row_major_strides(&shape, elem_size);
        match device {
            Device::Cpu => {
                let bytes = vec![0u8; total_bytes].into_boxed_slice();
                Ok(Tensor {
                    data: DataPtr::Cpu(bytes),
                    shape,
                    strides,
                    dtype,
                    device,
                })
            }
            Device::Cuda(dev_id) => {
                let stream = crate::cuda::get_stream()?;
                if stream.device_id != dev_id {
                    bail!(
                        "Stream device {} does not match target device {}",
                        stream.device_id,
                        dev_id
                    );
                }
                let gpu_mem = stream.inner().alloc_zeros::<u8>(total_bytes)?;
                Ok(Tensor {
                    data: DataPtr::Gpu(gpu_mem),
                    shape,
                    strides,
                    dtype,
                    device,
                })
            }
        }
    }

    pub fn from_string_literal(s: &str) -> Result<Self> {
        let (strings, shape, dtype_hint, device): (Vec<&str>, Vec<usize>, Option<DType>, Device) =
            parser::parse_full_tensor_string(s)?;
        let total_elements: usize = shape.iter().product();
        if strings.len() != total_elements {
            bail!(
                "Number of strings ({}) does not match shape product ({})",
                strings.len(),
                total_elements
            );
        }

        // 确定数据类型
        let dtype = if let Some(dt) = dtype_hint {
            dt
        } else {
            let all_int = strings
                .iter()
                .all(|s: &&str| !s.contains('.') && !s.contains('e') && !s.contains('E'));
            if all_int { DTYPE_INT32 } else { DTYPE_FLOAT32 }
        };

        let bytes = match dtype {
            DTYPE_FLOAT32 => {
                let mut data = Vec::with_capacity(total_elements);
                for s in &strings {
                    let val = s.parse::<f32>()?;
                    data.push(val);
                }
                bytemuck::cast_slice(&data).to_vec().into_boxed_slice()
            }
            DTYPE_INT32 => {
                let mut data = Vec::with_capacity(total_elements);
                for s in &strings {
                    let val = s.parse::<i32>()?;
                    data.push(val);
                }
                bytemuck::cast_slice(&data).to_vec().into_boxed_slice()
            }
            _ => bail!("Unsupported dtype: {}", dtype),
        };

        Self::new_from_bytes(bytes, shape, dtype, device)
    }
}
