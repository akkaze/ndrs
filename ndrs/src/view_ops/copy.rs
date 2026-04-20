//! 拷贝与连续化方法宏

#[macro_export]
macro_rules! impl_strided_copy_to {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn strided_copy_to(&self, dst: &mut Self) -> Result<(), String> {
            if self.shape != dst.shape {
                return Err(format!("Shape mismatch: self {:?}, dst {:?}", self.shape, dst.shape));
            }
            let src_tensor = $borrow(&self.handle);
            let mut dst_tensor = $borrow_mut(&dst.handle);
            if src_tensor.dtype() != dst_tensor.dtype() {
                return Err("Dtype mismatch".into());
            }
            if src_tensor.device() != dst_tensor.device() {
                return Err("Device mismatch".into());
            }
            let total_elements = self.shape.iter().product::<usize>();
            let elem_size = $crate::dtype::get_dtype_info(src_tensor.dtype()).unwrap().size;
            match src_tensor.device() {
                $crate::device::Device::CPU => unsafe {
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
                $crate::device::Device::GPU(_) => {
                    let ctx = src_tensor.cuda_ctx_ref().unwrap();
                    let stream = &ctx.stream;
                    let shape_dev = stream.clone_htod(&self.shape).map_err(|e| e.to_string())?;
                    let src_strides_dev = stream.clone_htod(&self.strides).map_err(|e| e.to_string())?;
                    let dst_strides_dev = stream.clone_htod(&dst.strides).map_err(|e| e.to_string())?;
                    let src_bytes = src_tensor.data_ptr(Some(stream));
                    let dst_bytes = dst_tensor.data_mut_ptr(Some(stream));
                    let stream_ptr = ctx.stream_ptr();
                    unsafe {
                        let err = $crate::kernel::gpu_strided_copy(
                            src_bytes,
                            self.offset,
                            src_strides_dev.device_ptr(stream).0 as *const usize,
                            self.shape.len() as i32,
                            shape_dev.device_ptr(stream).0 as *const usize,
                            dst_bytes,
                            dst.offset,
                            dst_strides_dev.device_ptr(stream).0 as *const usize,
                            elem_size,
                            total_elements,
                            stream_ptr,
                        );
                        if err != 0 {
                            return Err(format!("GPU strided copy failed: {}", err));
                        }
                    }
                    stream.synchronize().map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! impl_contiguous {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn contiguous(&self, out: &mut Self) -> Result<(), String> {
            if out.shape != self.shape {
                return Err("Output shape mismatch".into());
            }
            if !$borrow(&out.handle).is_contiguous() {
                return Err("Output must be contiguous".into());
            }
            self.strided_copy_to(out)
        }
    };
}
#[cfg(test)]
mod tests {
   
}