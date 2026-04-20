//! 形状操作方法宏（每个宏生成一个方法定义）

#[macro_export]
macro_rules! impl_broadcast_to {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
            if self.shape.len() > target_shape.len() {
                return Err("Cannot broadcast to fewer dimensions".into());
            }
            let mut new_strides = vec![0; target_shape.len()];
            let offset = target_shape.len() - self.shape.len();
            for i in 0..self.shape.len() {
                let target_dim = target_shape[offset + i];
                let self_dim = self.shape[i];
                if self_dim == target_dim {
                    new_strides[offset + i] = self.strides[i];
                } else if self_dim == 1 {
                    new_strides[offset + i] = 0;
                } else {
                    return Err(format!("Cannot broadcast dim {}", i));
                }
            }
            for i in 0..offset {
                new_strides[i] = 0;
            }
            Ok(self.as_strided(target_shape.to_vec(), new_strides, 0))
        }
    };
}

#[macro_export]
macro_rules! impl_transpose {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
            if axes.len() != self.shape.len() {
                return Err("Axes length mismatch".into());
            }
            let mut new_shape = Vec::with_capacity(self.shape.len());
            let mut new_strides = Vec::with_capacity(self.shape.len());
            let mut used = vec![false; self.shape.len()];
            for &axis in axes {
                if axis >= self.shape.len() || used[axis] {
                    return Err("Invalid or repeated axis".into());
                }
                used[axis] = true;
                new_shape.push(self.shape[axis]);
                new_strides.push(self.strides[axis]);
            }
            Ok(self.as_strided(new_shape, new_strides, self.offset))
        }
    };
}

#[macro_export]
macro_rules! impl_slice {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn slice(&self, info: &$crate::view::SliceInfo) -> Result<Self, String> {
            let slices = info.args();
            let mut new_offset = self.offset;
            let mut new_shape = Vec::with_capacity(slices.len());
            let mut new_strides = Vec::with_capacity(slices.len());
            for (dim, slice) in slices.iter().enumerate() {
                if dim >= self.shape.len() {
                    return Err("Too many slice dimensions".into());
                }
                let dim_size = self.shape[dim];
                let dim_stride = self.strides[dim];
                match slice {
                    $crate::view::SliceArg::Index(idx) => {
                        if *idx >= dim_size {
                            return Err("Index out of bounds".into());
                        }
                        new_offset += idx * dim_stride;
                    }
                    $crate::view::SliceArg::Range(start, end, step) => {
                        let start = *start;
                        let end = if *end == usize::MAX { dim_size } else { *end };
                        if start >= end || start >= dim_size {
                            return Err("Range out of bounds".into());
                        }
                        let len = (end - start + *step - 1) / *step;
                        new_shape.push(len);
                        new_strides.push(dim_stride * (*step as usize));
                        new_offset += start * dim_stride;
                    }
                    $crate::view::SliceArg::All => {
                        new_shape.push(dim_size);
                        new_strides.push(dim_stride);
                    }
                }
            }
            for dim in slices.len()..self.shape.len() {
                new_shape.push(self.shape[dim]);
                new_strides.push(self.strides[dim]);
            }
            Ok(self.as_strided(new_shape, new_strides, new_offset))
        }
    };
}

#[macro_export]
macro_rules! impl_concat_split {
    ($view_type:ident, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        fn concat_with_out(views: &[&Self], axis: usize, out: &mut Self) -> Result<(), String> {
            if views.is_empty() {
                return Err("No views to concatenate".into());
            }
            let first_shape = views[0].shape();
            if axis >= first_shape.len() {
                return Err("Axis out of bounds".into());
            }
            for v in views {
                if v.shape().len() != first_shape.len() {
                    return Err("All views must have same number of dimensions".into());
                }
                for d in 0..first_shape.len() {
                    if d != axis && v.shape()[d] != first_shape[d] {
                        return Err("All views must have same shape except on concat axis".into());
                    }
                }
            }
            let total_len: usize = views.iter().map(|v| v.shape()[axis]).sum();
            let mut expected_shape = first_shape.to_vec();
            expected_shape[axis] = total_len;
            if out.shape() != expected_shape {
                return Err("Output shape does not match concatenated shape".into());
            }
            let mut offset = 0;
            for view in views {
                let slice_len = view.shape()[axis];
                let mut slices = vec![$crate::view::SliceArg::All; first_shape.len()];
                slices[axis] = $crate::view::SliceArg::Range(offset, offset + slice_len, 1);
                let mut out_slice = out.slice(&$crate::view::SliceInfo::new(slices))?;
                out_slice.assign(view)?;
                offset += slice_len;
            }
            Ok(())
        }

        // 将当前张量沿指定轴分割成多个视图，输出到预先分配的 `out_views` 中。
        fn split_with_outs(
            &self,
            sizes: &[usize],
            axis: usize,
            out_views: &mut [Self],
        ) -> Result<(), String> {
            if sizes.len() != out_views.len() {
                return Err("Number of sizes does not match number of output views".into());
            }
            let total: usize = sizes.iter().sum();
            if self.shape()[axis] != total {
                return Err("Sum of sizes does not equal source size on axis".into());
            }
            let mut offset = 0;
            for (i, (&size, out_view)) in sizes.iter().zip(out_views.iter_mut()).enumerate() {
                let expected_shape = {
                    let mut shape = self.shape().to_vec();
                    shape[axis] = size;
                    shape
                };
                if out_view.shape() != expected_shape {
                    return Err(format!("Output view {} shape mismatch", i));
                }
                let mut slices = vec![$crate::view::SliceArg::All; self.shape().len()];
                slices[axis] = $crate::view::SliceArg::Range(offset, offset + size, 1);
                let src_slice = self.slice(&$crate::view::SliceInfo::new(slices))?;
                out_view.assign(&src_slice)?; // 修正方向
                offset += size;
            }
            Ok(())
        }

        fn concat(views: &[&Self], axis: usize) -> Result<Self, String> {
            if views.is_empty() {
                return Err("No views to concatenate".into());
            }
            let first_shape = views[0].shape();
            let total_len: usize = views.iter().map(|v| v.shape()[axis]).sum();
            let mut out_shape = first_shape.to_vec();
            out_shape[axis] = total_len;
            let out_tensor = $crate::tensor::Tensor::new_contiguous(out_shape, views[0].dtype())?;
            let mut out_view = Self::new($into_handle(out_tensor));
            Self::concat_with_out(views, axis, &mut out_view)?;
            Ok(out_view)
        }

        fn split(&self, sizes: &[usize], axis: usize) -> Result<Vec<Self>, String> {
            let total: usize = sizes.iter().sum();
            if self.shape()[axis] != total {
                return Err("Sum of sizes does not equal source size on axis".into());
            }
            let mut out_views = Vec::with_capacity(sizes.len());
            for &size in sizes {
                let mut shape = self.shape().to_vec();
                shape[axis] = size;
                let out_tensor = $crate::tensor::Tensor::new_contiguous(shape, self.dtype())?;
                out_views.push(Self::new($into_handle(out_tensor)));
            }
            self.split_with_outs(sizes, axis, &mut out_views)?;
            Ok(out_views)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::*;
    use ndrs_macros::s;

    #[test]
    fn test_rc_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        a_view.assign(&b_view).unwrap();
        assert_eq!(rc_view_to_vec_f32(&a_view), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_rc_broadcast_to() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let view = t.into_rc().as_view();
        let broadcasted = view.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcasted.shape(), &[3, 4]);
        assert_eq!(broadcasted.strides(), &[4, 0]);
    }

    #[test]
    fn test_rc_slice_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![10.0, 20.0], vec![1, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let mut sub = a_view.slice(&s![0..1, ..]).unwrap();
        sub.assign(&b_view).unwrap();
        assert_eq!(rc_view_to_vec_f32(&a_view), vec![10.0, 20.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rc_slice_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0], vec![1, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let mut sub = a_view.slice(&s![1..2, ..]).unwrap();
        sub += b_view;
        assert_eq!(rc_view_to_vec_f32(&a_view), vec![1.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn test_arc_broadcast_to() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let view = t.into_arc().as_view();
        let broadcasted = view.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcasted.shape(), &[3, 4]);
        assert_eq!(broadcasted.strides(), &[4, 0]);
    }

    #[test]
    fn test_arc_slice_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![10.0, 20.0], vec![1, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let mut sub = a_view.slice(&s![0..1, ..]).unwrap();
        sub.assign(&b_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&a_view), vec![10.0, 20.0, 3.0, 4.0]);
    }
    // 错误处理测试
    #[test]
    fn test_shape_mismatch_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0, 5.0], vec![3]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let result = a_view.assign(&b_view);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let view = t.into_rc().as_view();
        let result = view.slice(&s![3..5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::new_cpu_from_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let view = t.into_rc().as_view();
        let transposed = view.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[4, 12]);
        assert_eq!(
            rc_view_to_vec_f32(&transposed),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_T() {
        let t = Tensor::new_cpu_from_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let view = t.into_rc().as_view();
        let transposed = view.T().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[4, 12]);
        assert_eq!(
            rc_view_to_vec_f32(&transposed),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_transpose_3d() {
        let t = Tensor::new_cpu_from_f32((0..24).map(|x| x as f32).collect(), vec![2, 3, 4]);
        let view = t.into_rc().as_view();
        let transposed = view.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape(), &[4, 2, 3]);
        assert_eq!(transposed.strides(), &[4, 48, 16]);
    }

    #[test]
    fn test_concat_split() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();

        let c_view = RcTensorView::concat(&[&a_view, &b_view], 0).unwrap();
        assert_eq!(c_view.shape(), &[4]);
        assert_eq!(rc_view_to_vec_f32(&c_view), vec![1.0, 2.0, 3.0, 4.0]);

        let splits = c_view.split(&[2, 2], 0).unwrap();
        assert_eq!(splits.len(), 2);
        assert_eq!(rc_view_to_vec_f32(&splits[0]), vec![1.0, 2.0]);
        assert_eq!(rc_view_to_vec_f32(&splits[1]), vec![3.0, 4.0]);
    }
}
