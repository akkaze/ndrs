//! 加法操作宏（Add, AddAssign）

#[macro_export]
macro_rules! impl_add_for_view {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
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
                let a_dev = $borrow(&a.handle).device();
                let b_dev = $borrow(&b.handle).device();
                let out_dev = $borrow(&out.handle).device();
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
                let a_t = $borrow(&a.handle);
                let b_t = $borrow(&b.handle);
                let mut out_t = $borrow_mut(&out.handle);

                let n = a.shape().iter().product();
                let add_op = $crate::dtype::get_add_op(a_t.dtype()).expect("Add op not registered");
                match a_t.device() {
                    $crate::device::Device::CPU => {
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
                            $crate::device::Device::CPU,
                            None,
                        );
                    }
                    $crate::device::Device::GPU(_) => {
                        let ctx = a_t.cuda_ctx_ref().unwrap();
                        let stream = &ctx.stream;
                        let a_strides_dev =
                            stream.clone_htod(a.strides()).map_err(|e| e.to_string())?;
                        let b_strides_dev =
                            stream.clone_htod(b.strides()).map_err(|e| e.to_string())?;
                        let c_strides_dev = stream
                            .clone_htod(out.strides())
                            .map_err(|e| e.to_string())?;
                        let shape_dev = stream.clone_htod(a.shape()).map_err(|e| e.to_string())?;

                        let a_ptr = a_t.data_ptr(Some(stream));
                        let b_ptr = b_t.data_ptr(Some(stream));
                        let c_ptr = out_t.data_mut_ptr(Some(stream));
                        let stream_ptr = ctx.stream_ptr();

                        add_op(
                            a_ptr,
                            a_strides_dev.device_ptr(stream).0 as *const usize,
                            b_ptr,
                            b_strides_dev.device_ptr(stream).0 as *const usize,
                            c_ptr,
                            c_strides_dev.device_ptr(stream).0 as *const usize,
                            shape_dev.device_ptr(stream).0 as *const usize,
                            a.shape().len(),
                            n,
                            $crate::device::Device::GPU(0),
                            Some(stream_ptr),
                        );
                        stream.synchronize().map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndrs_macros::s;
    use crate::device::cuda_available;
    use crate::device::set_current_device;
    
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

    // GPU 测试（需要 CUDA 设备）
    #[test]
    fn test_gpu_add_basic() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();

        let mut a_gpu = a_view.clone();
        let mut b_gpu = b_view.clone();
        a_view.to_gpu(&mut a_gpu, 0).unwrap();
        b_view.to_gpu(&mut b_gpu, 0).unwrap();

        let out_shape = a_view.shape().to_vec();
        let elem_size = std::mem::size_of::<f32>();
        let total_bytes = out_shape.iter().product::<usize>() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            out_shape,
            a_view.dtype(),
        )
        .unwrap();
        let out_handle = out_tensor.into_arc();
        let out_view = out_handle.as_view();
        let mut out_gpu = out_view.clone();
        out_view.to_gpu(&mut out_gpu, 0).unwrap();

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
}
