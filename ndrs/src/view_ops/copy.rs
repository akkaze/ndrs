//! 拷贝与连续化方法宏
use crate::cuda;

#[macro_export]
macro_rules! impl_strided_copy_to {
    ($view_type:ident, $lock:ident, $into_handle:expr) => {
        fn strided_copy_to(&self, dst: &mut Self) -> Result<(), String> {
            if self.shape != dst.shape {
                return Err(format!(
                    "Shape mismatch: self {:?}, dst {:?}",
                    self.shape, dst.shape
                ));
            }
            let src_cell = $lock(&self.handle);
            let src_tensor = src_cell.borrow();
            let dst_cell = $lock(&dst.handle);
            let mut dst_tensor = dst_cell.borrow_mut();

            if src_tensor.dtype() != dst_tensor.dtype() {
                return Err("Dtype mismatch".into());
            }
            if src_tensor.device() != dst_tensor.device() {
                return Err("Device mismatch".into());
            }
            let total_elements = self.shape.iter().product::<usize>();
            let elem_size = get_dtype_info(src_tensor.dtype()).unwrap().size;
            match src_tensor.device() {
                Device::Cpu => unsafe {
                    cpu_strided_copy(
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
                    let stream = cuda::get_stream().map_err(|e| e.to_string())?;
                    let shape_dev = stream
                        .inner()
                        .clone_htod(&self.shape)
                        .map_err(|e| e.to_string())?;
                    let src_strides_dev = stream
                        .inner()
                        .clone_htod(&self.strides)
                        .map_err(|e| e.to_string())?;
                    let dst_strides_dev = stream
                        .inner()
                        .clone_htod(&dst.strides)
                        .map_err(|e| e.to_string())?;
                    let src_bytes = src_tensor.data_ptr(Some(stream.inner()));
                    let dst_bytes = dst_tensor.data_mut_ptr(Some(stream.inner()));
                    let stream_ptr = stream.as_ptr();
                    unsafe {
                        let err = gpu_strided_copy(
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
                            return Err(format!("GPU strided copy failed: {}", err));
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
    ($view_type:ident, $lock:ident, $into_handle:expr) => {
        fn contiguous(&self, out: &mut Self) -> Result<(), String> {
            if out.shape != self.shape {
                return Err("Output shape mismatch".into());
            }
            // 使用 $lock 获取可借用的对象，然后调用 .borrow() 进行只读检查
            if !$lock(&out.handle).borrow().is_contiguous() {
                return Err("Output must be contiguous".into());
            }
            self.strided_copy_to(out)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::{
        self, get_device_count as get_cuda_device_count, is_available as cuda_available,
        set_device as set_current_device,
    };
    use crate::s;
    use crate::tensor::Tensor;
    use crate::view::TensorViewOps;
    use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};
    use crate::DTYPE_FLOAT32;

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
