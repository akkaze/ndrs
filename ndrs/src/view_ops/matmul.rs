use anyhow::{Context, Result, anyhow, bail};
/// 矩阵乘法方法宏

#[macro_export]
macro_rules! impl_matmul_into {
    ($view_type:ident, $handle:ty) => {
        fn matmul_into(&self, other: &Self, out: &mut Self) -> anyhow::Result<()> {
            use anyhow::{Context, bail};

            let shape_self = self.shape();
            let shape_other = other.shape();
            let shape_out = out.shape();
            if shape_self.len() != 2 || shape_other.len() != 2 || shape_out.len() != 2 {
                bail!("matmul only supports 2D tensors");
            }
            let (m, k1) = (shape_self[0], shape_self[1]);
            let (k2, n) = (shape_other[0], shape_other[1]);
            if k1 != k2 {
                bail!("Inner dimensions must match");
            }
            if shape_out != &[m, n] {
                bail!("Output shape must be [M, N]");
            }
            let a_cell = self.handle.lock();
            let a_t = a_cell.borrow();
            let b_cell = other.handle.lock();
            let b_t = b_cell.borrow();
            let c_cell = out.handle.lock();
            let mut c_t = c_cell.borrow_mut();
            if a_t.dtype() != b_t.dtype() || a_t.dtype() != c_t.dtype() {
                bail!("Dtype mismatch");
            }
            if a_t.dtype() != $crate::DTYPE_FLOAT32 {
                bail!("matmul only supports f32 for now");
            }
            let a_strides = self.strides();
            let b_strides = other.strides();
            let c_strides = out.strides();
            let a_stride_row = a_strides[0];
            let a_stride_col = a_strides[1];
            let b_stride_row = b_strides[0];
            let b_stride_col = b_strides[1];
            let c_stride_row = c_strides[0];
            let c_stride_col = c_strides[1];
            let a_ptr = a_t.data_ptr(None);
            let b_ptr = b_t.data_ptr(None);
            let c_ptr = c_t.data_mut_ptr(None);
            match a_t.device() {
                $crate::device::Device::Cpu => unsafe {
                    $crate::kernel::cpu_matmul_strided_f32(
                        a_ptr as *const f32,
                        a_stride_row,
                        a_stride_col,
                        b_ptr as *const f32,
                        b_stride_row,
                        b_stride_col,
                        c_ptr as *mut f32,
                        c_stride_row,
                        c_stride_col,
                        m as i32,
                        n as i32,
                        k1 as i32,
                    );
                },
                $crate::device::Device::Cuda(_) => {
                    let stream = $crate::cuda::get_stream().context("Failed to get CUDA stream")?;
                    let stream_ptr = stream.as_ptr();
                    unsafe {
                        let err = $crate::kernel::gpu_matmul_strided_f32(
                            a_ptr as *const f32,
                            a_stride_row,
                            a_stride_col,
                            b_ptr as *const f32,
                            b_stride_row,
                            b_stride_col,
                            c_ptr as *mut f32,
                            c_stride_row,
                            c_stride_col,
                            m as i32,
                            n as i32,
                            k1 as i32,
                            stream_ptr,
                        );
                        if err != 0 {
                            bail!("GPU matmul failed with error {}", err);
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! impl_matmul {
    ($view_type:ident, $handle:ty) => {
        fn matmul(&self, other: &Self) -> anyhow::Result<Self> {
            let m = self.shape()[0];
            let n = other.shape()[1];
            let out_tensor =
                $crate::tensor::Tensor::new_contiguous(vec![m, n], self.dtype(), self.device())?;
            let mut out_view = Self::new(<$handle>::from_tensor(out_tensor));
            self.matmul_into(other, &mut out_view)?;
            Ok(out_view)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::view::rc_view_to_vec_f32;
    use crate::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let c_view = a_view.matmul(&b_view).unwrap();
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_eq!(rc_view_to_vec_f32(&c_view), expected);
    }
}
