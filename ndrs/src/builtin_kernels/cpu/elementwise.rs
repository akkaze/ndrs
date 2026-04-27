// ndrs/src/backend/cpu/elementwise.rs
use crate::dtype::{DTypeMapping, get_dtype_info};
use crate::view::{ArcTensorView, RcTensorView, TensorViewOps};
use anyhow::{Result, bail};

/// 为视图类型生成 CPU 逐元素运算的静态方法
macro_rules! impl_cpu_elementwise {
    ($view_type:ident) => {
        impl $view_type {
            /// 静态方法：对一组形状相同的视图执行逐元素运算。
            /// - `output`: 输出视图（可变）
            /// - `inputs`: 输入视图列表，形状必须与输出相同
            /// - `f`: 回调函数，参数为各输入视图在当前元素的值，返回输出值。
            pub fn elementwise<T, F>(output: &mut Self, inputs: &[&Self], mut f: F) -> Result<()>
            where
                T: bytemuck::Pod + DTypeMapping,
                F: FnMut(&[T]) -> T,
            {
                let shape = output.shape();
                let total = output.size();
                let dtype = T::DTYPE;

                if output.dtype() != dtype {
                    bail!("Output dtype mismatch");
                }
                for inp in inputs {
                    if inp.dtype() != dtype {
                        bail!("Input dtype mismatch");
                    }
                    if inp.shape() != shape {
                        bail!("Input shape mismatch");
                    }
                }

                let elem_size = std::mem::size_of::<T>();
                let out_ptr = unsafe { output.raw_data_ptr() } as *mut T;
                let out_strides = output.strides();

                // 预先获取每个输入的步长和基指针
                let mut input_data = Vec::with_capacity(inputs.len());
                for inp in inputs {
                    let ptr = unsafe { inp.raw_data_ptr() } as *const T;
                    let strides = inp.strides();
                    input_data.push((ptr, strides));
                }

                // 线性遍历所有元素，使用步长计算偏移
                for linear_idx in 0..total {
                    // 计算输出偏移（元素数）
                    let out_offset =
                        Self::linear_to_offset(linear_idx, shape, out_strides) / elem_size;
                    // 收集每个输入的值
                    let mut values = Vec::with_capacity(inputs.len());
                    for (ptr, strides) in &input_data {
                        let inp_offset =
                            Self::linear_to_offset(linear_idx, shape, strides) / elem_size;
                        let val = unsafe { *ptr.add(inp_offset) };
                        values.push(val);
                    }
                    let result = f(&values);
                    unsafe { *out_ptr.add(out_offset) = result };
                }
                Ok(())
            }

            // 辅助：将线性索引转换为字节偏移
            fn linear_to_offset(linear_idx: usize, shape: &[usize], strides: &[usize]) -> usize {
                let mut offset = 0;
                let mut rem = linear_idx;
                for d in (0..shape.len()).rev() {
                    let idx = rem % shape[d];
                    rem /= shape[d];
                    offset += idx * strides[d];
                }
                offset
            }
        }
    };
}

// 为两种视图生成实现
impl_cpu_elementwise!(RcTensorView);
impl_cpu_elementwise!(ArcTensorView);
