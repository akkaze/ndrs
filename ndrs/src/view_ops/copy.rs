use anyhow::{Context, Result, anyhow, bail};
/// 拷贝与连续化方法宏

#[macro_export]
macro_rules! impl_strided_copy_to {
    ($view_type:ident, $handle:ty) => {
        fn strided_copy_to(&self, dst: &mut Self) -> anyhow::Result<()> {
            use anyhow::{Context, bail};

            if self.shape != dst.shape {
                bail!("Shape mismatch: self {:?}, dst {:?}", self.shape, dst.shape);
            }
            let src_cell = self.handle.lock();
            let src_tensor = src_cell.borrow();
            let dst_cell = dst.handle.lock();
            let mut dst_tensor = dst_cell.borrow_mut();

            if src_tensor.dtype() != dst_tensor.dtype() {
                bail!("Dtype mismatch");
            }
            if src_tensor.device() != dst_tensor.device() {
                bail!("Device mismatch");
            }
            let total_elements = self.shape.iter().product::<usize>();
            let elem_size = get_dtype_info(src_tensor.dtype()).unwrap().size;
            match src_tensor.device() {
                Device::Cpu => unsafe {
                    $crate::kernel::cpu_strided_copy(
                        src_tensor.data_ptr(None),
                        self.offset,
                        self.strides.as_ptr(),
                        self.shape.len() as i32,
                        self.shape.as_ptr(),
                        dst_tensor.data_mut_ptr(None),
                        dst.offset,
                        dst.strides.as_ptr(),
                        elem_size,
                        total_elements,
                    );
                },
                Device::Cuda(_) => {
                    let stream = $crate::cuda::get_stream().context("Failed to get CUDA stream")?;
                    let shape_dev = stream
                        .inner()
                        .clone_htod(&self.shape)
                        .context("Failed to copy shape to device")?;
                    let src_strides_dev = stream
                        .inner()
                        .clone_htod(&self.strides)
                        .context("Failed to copy src strides to device")?;
                    let dst_strides_dev = stream
                        .inner()
                        .clone_htod(&dst.strides)
                        .context("Failed to copy dst strides to device")?;
                    let src_bytes = src_tensor.data_ptr(Some(stream.inner()));
                    let dst_bytes = dst_tensor.data_mut_ptr(Some(stream.inner()));
                    let stream_ptr = stream.as_ptr();
                    unsafe {
                        let err = $crate::kernel::gpu_strided_copy(
                            src_bytes,
                            self.offset,
                            src_strides_dev.device_ptr(stream.inner()).0 as *const usize,
                            self.shape.len() as i32,
                            shape_dev.device_ptr(stream.inner()).0 as *const usize,
                            dst_bytes,
                            dst.offset,
                            dst_strides_dev.device_ptr(stream.inner()).0 as *const usize,
                            elem_size,
                            total_elements,
                            stream_ptr,
                        );
                        if err != 0 {
                            bail!("GPU strided copy failed: {}", err);
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! impl_contiguous {
    ($view_type:ident, $handle:ty) => {
        fn contiguous_into(&self, out: &mut Self) -> anyhow::Result<()> {
            use anyhow::bail;

            if out.shape != self.shape {
                bail!("Output shape mismatch");
            }
            if !out.handle.lock().borrow().is_contiguous() {
                bail!("Output must be contiguous");
            }
            self.strided_copy_to(out)
        }

        fn contiguous(&self) -> anyhow::Result<Self::Handle> {
            let out_tensor = $crate::tensor::Tensor::new_contiguous(
                self.shape().to_vec(),
                self.dtype(),
                self.device(),
            )?;
            let mut out_view = Self::new(<$handle>::from_tensor(out_tensor));
            self.contiguous_into(&mut out_view)?;
            Ok(out_view.into_handle())
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DTYPE_FLOAT32;
    use crate::cuda;
    use crate::cuda::{
        get_device_count as get_cuda_device_count, is_available as cuda_available,
        set_device as set_current_device,
    };
    use crate::s;
    use crate::tensor::Tensor;
    use crate::view::TensorViewOps;
    use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};

    #[test]
    fn test_arc_same_device_copy() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.strided_copy_to(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_strided_copy_to() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.strided_copy_to(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }
}
