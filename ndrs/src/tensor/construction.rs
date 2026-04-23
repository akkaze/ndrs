use super::*;
use crate::device::Device;
use crate::dtype::{get_dtype_info, DType, DTypeMapping};
use crate::tensor::{DataPtr, Tensor};
use crate::{DTYPE_FLOAT32, DTYPE_INT32};
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
            device: Device::Cpu,
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
            device: Device::Cpu,
        })
    }

    /// `dtype_hint` 为 `None` 时自动推断：所有字符串可解析为整数则 `DTYPE_INT32`，否则 `DTYPE_FLOAT32`。
    /// 若字符串列表为空且形状元素个数 > 0，则创建零张量。
    pub fn from_strings(
        strings: &[&str],
        shape: &[usize],
        dtype_hint: Option<&str>,
    ) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();

        // 空数据但形状非零 -> 零张量
        if strings.is_empty() && total_elements > 0 {
            let dtype = match dtype_hint {
                Some("i32") => DTYPE_INT32,
                Some("f32") => DTYPE_FLOAT32,
                _ => DTYPE_FLOAT32, // 默认
            };
            return Self::new_contiguous(shape.to_vec(), dtype);
        }

        // 数据数量必须匹配形状
        if strings.len() != total_elements {
            return Err(format!(
                "Number of strings ({}) does not match shape product ({})",
                strings.len(),
                total_elements
            ));
        }

        // 确定数据类型
        let dtype = match dtype_hint {
            Some("i32") => DTYPE_INT32,
            Some("f32") => DTYPE_FLOAT32,
            None => {
                let all_int = strings
                    .iter()
                    .all(|s| !s.contains('.') && !s.contains('e') && !s.contains('E'));
                if all_int {
                    DTYPE_INT32
                } else {
                    DTYPE_FLOAT32
                }
            }
            Some(other) => return Err(format!("Unsupported dtype hint: {}", other)),
        };

        match dtype {
            DTYPE_FLOAT32 => {
                let mut data = Vec::with_capacity(strings.len());
                for s in strings {
                    let val = s
                        .parse::<f32>()
                        .map_err(|e| format!("Failed to parse '{}' as f32: {}", s, e))?;
                    data.push(val);
                }
                Ok(Self::new_cpu_from_f32(data, shape.to_vec()))
            }
            DTYPE_INT32 => {
                let mut data = Vec::with_capacity(strings.len());
                for s in strings {
                    let val = s
                        .parse::<i32>()
                        .map_err(|e| format!("Failed to parse '{}' as i32: {}", s, e))?;
                    data.push(val);
                }
                Ok(Self::new_cpu_from_i32(data, shape.to_vec()))
            }
            _ => Err(format!("Unsupported dtype: {}", dtype)),
        }
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
            device: Device::Cpu,
        })
    }
}
