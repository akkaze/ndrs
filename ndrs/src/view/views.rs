//! 具体视图类型：RcTensorView 和 ArcTensorView

use super::slice::{SliceArg, SliceInfo};
use super::trait_def::TensorViewOps;
use crate::device::{get_or_create_context, Device};
use crate::dtype::{get_dtype_info, DType};
use crate::kernel::*;
use crate::tensor::Tensor;
use cudarc::driver::DevicePtr;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// 辅助锁函数
fn lock_rc(handle: &Rc<RefCell<Tensor>>) -> std::cell::Ref<Tensor> {
    handle.borrow()
}
fn lock_rc_mut(handle: &Rc<RefCell<Tensor>>) -> std::cell::RefMut<Tensor> {
    handle.borrow_mut()
}
fn lock_arc(handle: &Arc<Mutex<Tensor>>) -> std::sync::MutexGuard<Tensor> {
    handle.lock().unwrap()
}
fn lock_arc_mut(handle: &Arc<Mutex<Tensor>>) -> std::sync::MutexGuard<Tensor> {
    handle.lock().unwrap()
}
fn into_rc(t: Tensor) -> Rc<RefCell<Tensor>> {
    t.into_rc()
}
fn into_arc(t: Tensor) -> Arc<Mutex<Tensor>> {
    t.into_arc()
}

macro_rules! impl_tensor_view {
    ($name:ident, $handle:ty, $borrow:ident, $borrow_mut:ident, $into_handle:expr) => {
        #[derive(Clone)]
        pub struct $name {
            handle: $handle,
            offset: usize,
            shape: Vec<usize>,
            strides: Vec<usize>,
        }

        impl $name {
            pub fn new(handle: $handle) -> Self {
                let (shape, strides) = {
                    let tensor = $borrow(&handle);
                    (tensor.shape().to_vec(), tensor.strides().to_vec())
                };
                $name {
                    handle,
                    offset: 0,
                    shape,
                    strides,
                }
            }

            pub fn into_handle(self) -> $handle {
                self.handle
            }
            pub fn handle(&self) -> &$handle {
                &self.handle
            }

            // 根据当前视图的设备创建相同形状的空输出张量
            fn create_output(&self) -> Result<Self, String> {
                let tensor = $borrow(&self.handle);
                let elem_size = get_dtype_info(tensor.dtype()).unwrap().size;
                let total_bytes = self.size() * elem_size;
                let shape = self.shape().to_vec();
                let dtype = tensor.dtype();
                let device = tensor.device();
                let cuda_ctx = tensor.cuda_ctx_ref().cloned();

                let new_tensor = match device {
                    Device::CPU => {
                        let bytes = vec![0u8; total_bytes].into_boxed_slice();
                        Tensor::new_cpu_from_bytes(bytes, shape, dtype)?
                    }
                    Device::GPU(dev_id) => {
                        let ctx = match cuda_ctx {
                            Some(ctx) => ctx,
                            None => get_or_create_context(dev_id)?,
                        };
                        let bytes = vec![0u8; total_bytes];
                        let gpu_mem = ctx
                            .stream
                            .clone_htod::<u8, _>(&bytes)
                            .map_err(|e| e.to_string())?;
                        let mut t =
                            Tensor::new_cpu_from_bytes(bytes.into_boxed_slice(), shape, dtype)?;
                        t.data = crate::tensor::DataPtr::Gpu(gpu_mem);
                        t.device = device;
                        t.cuda_ctx = Some(ctx);
                        t
                    }
                };
                Ok($name::new($into_handle(new_tensor)))
            }
        }

        impl TensorViewOps for $name {
            type Handle = $handle;

            fn new(handle: $handle) -> Self {
                Self::new(handle)
            }

            fn as_strided(
                &self,
                new_shape: Vec<usize>,
                new_strides: Vec<usize>,
                offset: usize,
            ) -> Self {
                assert_eq!(new_shape.len(), new_strides.len());
                $name {
                    handle: self.handle.clone(),
                    offset: self.offset + offset,
                    shape: new_shape,
                    strides: new_strides,
                }
            }

            // 设备间传输
            fn to(&self, out: &mut Self, target_device: Device) -> Result<(), String> {
                if self.shape != out.shape {
                    return Err("Shape mismatch".into());
                }
                // 同一句柄死锁预防
                if std::ptr::eq(&*self.handle, &*out.handle) {
                    let mut temp = self.create_output()?;
                    self.strided_copy_to(&mut temp)?;
                    return temp.to(out, target_device);
                }
                let src_t = $borrow(&self.handle);
                let mut dst_t = $borrow_mut(&out.handle);
                if src_t.dtype() != dst_t.dtype() {
                    return Err("Dtype mismatch".into());
                }
                match (src_t.device(), target_device) {
                    (a, b) if a == b => {
                        drop(src_t);
                        drop(dst_t);
                        self.strided_copy_to(out)
                    }
                    (Device::CPU, Device::GPU(idx)) => {
                        let ctx = get_or_create_context(idx)?;
                        let bytes = match &src_t.data {
                            crate::tensor::DataPtr::Cpu(b) => b.as_ref(),
                            _ => unreachable!(),
                        };
                        let gpu_mem = ctx
                            .stream
                            .clone_htod::<u8, _>(bytes)
                            .map_err(|e| e.to_string())?;
                        dst_t.data = crate::tensor::DataPtr::Gpu(gpu_mem);
                        dst_t.device = Device::GPU(idx);
                        dst_t.cuda_ctx = Some(ctx);
                        Ok(())
                    }
                    (Device::GPU(_), Device::CPU) => {
                        let gpu_slice = match &src_t.data {
                            crate::tensor::DataPtr::Gpu(s) => s,
                            _ => unreachable!(),
                        };
                        let bytes = src_t
                            .cuda_ctx_ref()
                            .ok_or("Missing CUDA context")?
                            .stream
                            .clone_dtoh(gpu_slice)
                            .map_err(|e| e.to_string())?;
                        dst_t.data = crate::tensor::DataPtr::Cpu(bytes.into_boxed_slice());
                        dst_t.device = Device::CPU;
                        dst_t.cuda_ctx = None;
                        Ok(())
                    }
                    _ => Err("Unsupported device conversion".into()),
                }
            }

            fn shape(&self) -> &[usize] {
                &self.shape
            }

            fn strides(&self) -> &[usize] {
                &self.strides
            }

            fn offset(&self) -> usize {
                self.offset
            }

            fn dtype(&self) -> DType {
                $borrow(&self.handle).dtype()
            }

            fn size(&self) -> usize {
                self.shape.iter().product()
            }

            fn assign(&mut self, src: &Self) -> Result<(), String> {
                if self.shape != src.shape {
                    return Err("Shape mismatch".into());
                }
                src.strided_copy_to(self)
            }

            // 形状操作方法
            $crate::impl_broadcast_to!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_transpose!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_slice!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_concat_split!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_strided_copy_to!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_contiguous!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_matmul_with_out!($name, $borrow, $borrow_mut, $into_handle);
            $crate::impl_matmul!($name, $borrow, $borrow_mut, $into_handle);
        }

        // 加法操作（Add, AddAssign）
        $crate::impl_add_for_view!($name, $borrow, $borrow_mut, $into_handle);
    };
}

impl_tensor_view!(
    RcTensorView,
    Rc<RefCell<Tensor>>,
    lock_rc,
    lock_rc_mut,
    into_rc
);
impl_tensor_view!(
    ArcTensorView,
    Arc<Mutex<Tensor>>,
    lock_arc,
    lock_arc_mut,
    into_arc
);

pub trait AsView {
    type View: TensorViewOps;
    fn as_view(&self) -> Self::View;
}

impl AsView for Rc<RefCell<Tensor>> {
    type View = RcTensorView;
    fn as_view(&self) -> Self::View {
        RcTensorView::new(self.clone())
    }
}

impl AsView for Arc<Mutex<Tensor>> {
    type View = ArcTensorView;
    fn as_view(&self) -> Self::View {
        ArcTensorView::new(self.clone())
    }
}

/// 为视图类型生成转换为 `Vec<T>` 的辅助函数
macro_rules! define_view_to_vec {
    ($func_name:ident, $view_type:ident, $into_handle:expr, $lock_method:ident, $copy_method:ident) => {
        fn $func_name<T: bytemuck::Pod + crate::dtype::DTypeMapping>(view: &$view_type) -> Vec<T> {
            let elem_size = std::mem::size_of::<T>();
            let total_bytes = view.size() * elem_size;
            let out_tensor = crate::tensor::Tensor::new_cpu_from_bytes(
                vec![0u8; total_bytes].into_boxed_slice(),
                view.shape().to_vec(),
                T::DTYPE,
            )
            .expect("Failed to create output tensor");
            let out_handle = $into_handle(out_tensor);
            let mut out_view = $view_type::new(out_handle);
            view.$copy_method(&mut out_view)
                .expect("Failed to copy data");
            let tensor = $lock_method(&out_view.handle);
            let bytes = tensor.as_bytes().expect("Failed to get bytes");
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, view.size()).to_vec() }
        }
    };
}

// 为两种视图类型生成函数
define_view_to_vec!(rc_view_to_vec, RcTensorView, into_rc, lock_rc, to_cpu);
define_view_to_vec!(arc_view_to_vec, ArcTensorView, into_arc, lock_arc, to_cpu);

// 可选：为常用类型提供便捷别名
pub fn rc_view_to_vec_f32(view: &RcTensorView) -> Vec<f32> {
    rc_view_to_vec(view)
}
pub fn arc_view_to_vec_f32(view: &ArcTensorView) -> Vec<f32> {
    arc_view_to_vec(view)
}

// 原有单元测试保留在此（已适配新路径）
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{self, cuda_available, get_cuda_device_count, set_current_device};
    use crate::s;
    use crate::tensor::Tensor;
    use crate::view::trait_def::TensorViewOps;
    use crate::DTYPE_FLOAT32;

    // ---------- RcTensorView 测试 ----------
    #[test]
    fn test_rc_view_creation() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = t.into_rc().as_view();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[8, 4]);
        assert_eq!(view.offset(), 0);
    }

    // ---------- ArcTensorView 测试 ----------
    #[test]
    fn test_arc_view_creation() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = t.into_arc().as_view();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[8, 4]);
        assert_eq!(view.offset(), 0);
    }

    #[test]
    fn test_rc_contiguous() {
        let t = Tensor::new_cpu_from_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let view = t.into_rc().as_view();
        let transposed = view.as_strided(vec![3, 2], vec![4, 12], 0);
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; 6 * 4].into_boxed_slice(),
            vec![3, 2],
            DTYPE_FLOAT32,
        )
        .unwrap();
        let out_handle = out_tensor.into_rc();
        let mut out_view = RcTensorView::new(out_handle);
        transposed.contiguous(&mut out_view).unwrap();
        assert_eq!(
            rc_view_to_vec_f32(&out_view),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_rc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_rc().as_view();
        let mut dst_view = dst.into_rc().as_view();
        src_view.to_cpu(&mut dst_view).unwrap();
        assert_eq!(rc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_contiguous() {
        let t = Tensor::new_cpu_from_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let view = t.into_arc().as_view();
        let transposed = view.as_strided(vec![3, 2], vec![4, 12], 0);
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; 6 * 4].into_boxed_slice(),
            vec![3, 2],
            DTYPE_FLOAT32,
        )
        .unwrap();
        let out_handle = out_tensor.into_arc();
        let mut out_view = ArcTensorView::new(out_handle);
        transposed.contiguous(&mut out_view).unwrap();
        assert_eq!(
            arc_view_to_vec_f32(&out_view),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_arc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.to_cpu(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_slice_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0], vec![1, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let mut sub = a_view.slice(&s![1..2, ..]).unwrap();
        sub += b_view;
        assert_eq!(arc_view_to_vec_f32(&a_view), vec![1.0, 2.0, 8.0, 10.0]);
    }

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

    #[test]
    fn test_gpu_to_cpu_transfer() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let src_tensor = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let src_view = src_tensor.into_arc().as_view();

        let elem_size = std::mem::size_of::<f32>();
        let total_bytes = src_view.size() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            src_view.shape().to_vec(),
            src_view.dtype(),
        )
        .unwrap();
        let mut gpu_view = out_tensor.into_arc().as_view();
        src_view.to_gpu(&mut gpu_view, 0).unwrap();
        assert_eq!(gpu_view.handle().lock().unwrap().device(), Device::GPU(0));

        let back_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            src_view.shape().to_vec(),
            src_view.dtype(),
        )
        .unwrap();
        let mut back_cpu = back_tensor.into_arc().as_view();
        gpu_view.to_cpu(&mut back_cpu).unwrap();
        assert_eq!(arc_view_to_vec_f32(&back_cpu), vec![1.0, 2.0, 3.0]);
    }
}
