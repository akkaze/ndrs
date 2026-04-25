//! 加法操作宏（Add, AddAssign）

#[macro_export]
macro_rules! impl_add_for_view {
    ($view_type:ident, $handle:ty) => {
        impl std::ops::Add for $view_type {
            type Output = Self;
            fn add(self, other: Self) -> Self::Output {
                let target_shape = $crate::view::broadcast_shapes(self.shape(), other.shape())
                    .expect("Incompatible shapes for broadcast");
                let a_bcast = self.broadcast_to(&target_shape).unwrap();
                let b_bcast = other.broadcast_to(&target_shape).unwrap();
                let mut out = a_bcast.create_output().unwrap();
                $view_type::add(&a_bcast, &b_bcast, &mut out).unwrap();
                out
            }
        }

        impl std::ops::AddAssign for $view_type {
            fn add_assign(&mut self, other: Self) {
                assert_eq!(
                    self.shape(),
                    other.shape(),
                    "Shapes must match for AddAssign"
                );
                let mut temp = self.create_output().expect("Failed to create temp");
                self.strided_copy_to(&mut temp)
                    .expect("Copy to temp failed");
                let mut result = self.create_output().expect("Failed to create result");
                $view_type::add(&temp, &other, &mut result).expect("Addition failed");
                result.strided_copy_to(self).expect("Copy back failed");
            }
        }

        impl $view_type {
            pub fn add(a: &Self, b: &Self, out: &mut Self) -> Result<(), String> {
                let a_cell = a.handle.lock();
                let a_t = a_cell.borrow();
                let b_cell = b.handle.lock();
                let b_t = b_cell.borrow();
                let out_cell = out.handle.lock();
                let mut out_t = out_cell.borrow_mut();

                let a_dev = a_t.device();
                let b_dev = b_t.device();
                let out_dev = out_t.device();
                debug_assert!(
                    a_dev == b_dev && a_dev == out_dev,
                    "Add: all tensors must be on same device (a={:?}, b={:?}, out={:?})",
                    a_dev,
                    b_dev,
                    out_dev
                );

                if a.shape != b.shape || a.shape != out.shape {
                    return Err("Shape mismatch in add".into());
                }

                let n = a.shape().iter().product();
                let add_op = $crate::dtype::get_add_op(a_t.dtype(), a_dev).ok_or_else(|| {
                    format!(
                        "Add op not registered for dtype {} on device {:?}",
                        a_t.dtype(),
                        a_dev
                    )
                })?;
                match a_t.device() {
                    $crate::device::Device::Cpu => {
                        let a_ptr = a_t.data_ptr(None);
                        let b_ptr = b_t.data_ptr(None);
                        let c_ptr = out_t.data_mut_ptr(None);
                        add_op(
                            a_ptr,
                            a.strides().as_ptr(),
                            b_ptr,
                            b.strides().as_ptr(),
                            c_ptr,
                            out.strides().as_ptr(),
                            a.shape().as_ptr(),
                            a.shape().len(),
                            n,
                            $crate::device::Device::Cpu,
                            None,
                        );
                    }
                    $crate::device::Device::Cuda(_) => {
                        let stream = cuda::get_stream().map_err(|e| e.to_string())?;
                        let a_strides_dev = stream
                            .inner()
                            .clone_htod(a.strides())
                            .map_err(|e| e.to_string())?;
                        let b_strides_dev = stream
                            .inner()
                            .clone_htod(b.strides())
                            .map_err(|e| e.to_string())?;
                        let c_strides_dev = stream
                            .inner()
                            .clone_htod(out.strides())
                            .map_err(|e| e.to_string())?;
                        let shape_dev = stream
                            .inner()
                            .clone_htod(a.shape())
                            .map_err(|e| e.to_string())?;

                        let a_ptr = a_t.data_ptr(Some(stream.inner()));
                        let b_ptr = b_t.data_ptr(Some(stream.inner()));
                        let c_ptr = out_t.data_mut_ptr(Some(stream.inner()));
                        let stream_ptr = stream.as_ptr();

                        add_op(
                            a_ptr,
                            a_strides_dev.device_ptr(stream.inner()).0 as *const usize,
                            b_ptr,
                            b_strides_dev.device_ptr(stream.inner()).0 as *const usize,
                            c_ptr,
                            c_strides_dev.device_ptr(stream.inner()).0 as *const usize,
                            shape_dev.device_ptr(stream.inner()).0 as *const usize,
                            a.shape().len(),
                            n,
                            $crate::device::Device::Cuda(0),
                            Some(stream_ptr),
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::cuda;
    use crate::cuda::is_available;
    use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};

    use crate::*;

    #[test]
    fn test_rc_add() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let c_view = a_view + b_view;
        assert_eq!(rc_view_to_vec_f32(&c_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_rc_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        a_view += b_view;
        assert_eq!(rc_view_to_vec_f32(&a_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_arc_slice() {
        let t = Tensor::new_cpu_from_f32((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let view = t.into_arc().as_view();
        let sub = view.slice(&s![1..3, 2..4]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        let expected = vec![6.0, 7.0, 10.0, 11.0];
        assert_eq!(arc_view_to_vec_f32(&sub), expected);
    }

    #[test]
    fn test_arc_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        a_view.assign(&b_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&a_view), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_arc_add() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let c_view = a_view + b_view;
        assert_eq!(arc_view_to_vec_f32(&c_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_arc_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        a_view += b_view;
        assert_eq!(arc_view_to_vec_f32(&a_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_gpu_add_basic() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0);

        let a_cpu = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b_cpu = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let a_gpu = a_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let b_gpu = b_cpu.into_arc().as_view().to_gpu(0).unwrap();

        // 创建输出张量（全零）并上传到 GPU
        let zero_cpu = Tensor::new_contiguous(vec![2, 2], DTYPE_FLOAT32).unwrap();
        let mut out_gpu = zero_cpu.into_arc().as_view().to_gpu(0).unwrap();

        // GPU 加法
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        let result = arc_view_to_vec_f32(&out_gpu);
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(result, expected);
    }
    #[test]
    fn test_add_broadcast() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = Tensor::new_cpu_from_f32(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();

        let target_shape = crate::view::broadcast_shapes(a_view.shape(), b_view.shape()).unwrap();
        assert_eq!(target_shape, vec![3, 4]);

        let a_bcast = a_view.broadcast_to(&target_shape).unwrap();
        let b_bcast = b_view.broadcast_to(&target_shape).unwrap();

        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; 3 * 4 * 4].into_boxed_slice(),
            vec![3, 4],
            a_view.dtype(),
        )
        .unwrap();
        let out_handle = out_tensor.into_rc();
        let mut out_view = out_handle.as_view();
        RcTensorView::add(&a_bcast, &b_bcast, &mut out_view).unwrap();

        let result = rc_view_to_vec_f32(&out_view);
        let expected: Vec<f32> = (0..3)
            .flat_map(|i| (0..4).map(move |j| (i + 1) as f32 + (j + 4) as f32))
            .collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_async_gpu_add() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0);

        let a_cpu = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b_cpu = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let a_gpu = a_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let b_gpu = b_cpu.into_arc().as_view().to_gpu(0).unwrap();

        let zero_cpu = Tensor::new_contiguous(vec![2, 2], DTYPE_FLOAT32).unwrap();
        let mut out_gpu = zero_cpu.into_arc().as_view().to_gpu(0).unwrap();

        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        // 同步：拷贝回 CPU 即完成同步
        let result = arc_view_to_vec_f32(&out_gpu);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    // 第二个测试同样简化

    #[test]
    fn test_gpu_add_async_no_sync() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0);

        let a_cpu = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b_cpu = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let a_gpu = a_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let b_gpu = b_cpu.into_arc().as_view().to_gpu(0).unwrap();

        let zero_cpu = Tensor::new_contiguous(vec![2, 2], DTYPE_FLOAT32).unwrap();
        let mut out_gpu = zero_cpu.into_arc().as_view().to_gpu(0).unwrap();

        // 异步加法
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        let result = arc_view_to_vec_f32(&out_gpu);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_direct_add_and_add_assign() {
        use crate::s;
        use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};

        // 1. RcTensorView: Add
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let c_view = a_view + b_view;
        assert_eq!(rc_view_to_vec_f32(&c_view), vec![6.0, 8.0, 10.0, 12.0]);

        // 2. RcTensorView: AddAssign
        let mut a2 = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b2 = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a2_view = a2.into_rc().as_view();
        let b2_view = b2.into_rc().as_view();
        a2_view += b2_view;
        assert_eq!(rc_view_to_vec_f32(&a2_view), vec![6.0, 8.0, 10.0, 12.0]);

        // 3. ArcTensorView: Add
        let a3 = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b3 = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let a3_view = a3.into_arc().as_view();
        let b3_view = b3.into_arc().as_view();
        let c3_view = a3_view + b3_view;
        assert_eq!(arc_view_to_vec_f32(&c3_view), vec![6.0, 8.0, 10.0, 12.0]);

        // 4. ArcTensorView: AddAssign
        let mut a4 = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b4 = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a4_view = a4.into_arc().as_view();
        let b4_view = b4.into_arc().as_view();
        a4_view += b4_view;
        assert_eq!(arc_view_to_vec_f32(&a4_view), vec![6.0, 8.0, 10.0, 12.0]);

        // 5. 广播加法
        let a5 = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b5 = Tensor::new_cpu_from_f32(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
        let a5_view = a5.into_rc().as_view();
        let b5_view = b5.into_rc().as_view();
        let c5_view = a5_view + b5_view;
        let expected: Vec<f32> = (0..3)
            .flat_map(|i| (0..4).map(move |j| (i + 1) as f32 + (j + 4) as f32))
            .collect();
        assert_eq!(rc_view_to_vec_f32(&c5_view), expected);

        // 6. 切片上的 AddAssign
        let mut a6 = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b6 = Tensor::new_cpu_from_f32(vec![10.0, 20.0], vec![1, 2]);
        let mut a6_view = a6.into_rc().as_view();
        let b6_view = b6.into_rc().as_view();
        let mut sub = a6_view.slice(&s![0..1, ..]).unwrap();
        sub += b6_view;
        assert_eq!(rc_view_to_vec_f32(&a6_view), vec![11.0, 22.0, 3.0, 4.0]);
    }
}
