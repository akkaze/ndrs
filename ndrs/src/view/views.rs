//! 具体视图类型：RcTensorView 和 ArcTensorView
use super::slice::{SliceArg, SliceInfo};
use super::trait_def::TensorViewOps;
use crate::cuda;
use crate::cuda::Stream;
use crate::dtype::{get_dtype_info, DType};
use crate::kernel::*;
use crate::tensor::{ArcTensor, DataPtr, RcTensor, Tensor};
use crate::Device;
use cudarc::driver::DevicePtr;
use parking_lot::ReentrantMutexGuard;
use std::cell::RefCell;
use std::cell::{Ref, RefMut};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// Rc<RefCell<Tensor>> 的锁：返回 &RefCell<Tensor>
fn lock_rc(handle: &RcTensor) -> &RefCell<Tensor> {
    &*handle.0
}

// Arc<ReentrantMutex<RefCell<Tensor>>> 的锁：返回 ReentrantMutexGuard<RefCell<Tensor>>
fn lock_arc(handle: &ArcTensor) -> parking_lot::ReentrantMutexGuard<RefCell<Tensor>> {
    handle.0.lock()
}

fn into_rc(t: Tensor) -> RcTensor {
    RcTensor::from(t)
}
fn into_arc(t: Tensor) -> ArcTensor {
    ArcTensor::from(t)
}

macro_rules! impl_tensor_view {
    ($name:ident, $handle:ty, $lock:ident, $into_handle:expr) => {
        #[derive(Clone)]
        pub struct $name {
            handle: $handle,
            offset: usize,
            shape: Vec<usize>,
            strides: Vec<usize>,
            dtype: DType,
            device: Device,
        }

        impl $name {
            pub fn new(handle: $handle) -> Self {
                let (shape, strides, dtype, device) = {
                    let cell = $lock(&handle);
                    let tensor = cell.borrow(); // 只读借用
                    (
                        tensor.shape().to_vec(),
                        tensor.strides().to_vec(),
                        tensor.dtype(),
                        tensor.device(),
                    )
                };
                $name {
                    handle,
                    offset: 0,
                    shape,
                    strides,
                    dtype,
                    device,
                }
            }

            pub fn into_handle(self) -> $handle {
                self.handle
            }

            pub fn handle(&self) -> &$handle {
                &self.handle
            }

            fn create_output(&self) -> Result<Self, String> {
                self.create_output_on_device(self.device)
            }

            fn create_output_on_device(&self, device: Device) -> Result<Self, String> {
                let elem_size = get_dtype_info(self.dtype).unwrap().size;
                let total_bytes = self.size() * elem_size;
                let shape = self.shape().to_vec();
                let dtype = self.dtype;

                let new_tensor = match device {
                    Device::Cpu => {
                        let bytes = vec![0u8; total_bytes].into_boxed_slice();
                        Tensor::new_cpu_from_bytes(bytes, shape, dtype)?
                    }
                    Device::Cuda(dev_id) => {
                        let stream = cuda::get_stream().map_err(|e| e.to_string())?;
                        if stream.device_id != dev_id {
                            return Err(format!(
                                "Stream device {} does not match target device {}",
                                stream.device_id, dev_id
                            ));
                        }
                        let gpu_mem = stream
                            .inner()
                            .alloc_zeros::<u8>(total_bytes)
                            .map_err(|e| e.to_string())?;
                        let strides = Tensor::compute_row_major_strides(&shape, elem_size);
                        Tensor {
                            data: DataPtr::Gpu(gpu_mem),
                            shape,
                            strides,
                            dtype,
                            device,
                        }
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
                    dtype: self.dtype,
                    device: self.device,
                }
            }

            fn is_contiguous(&self) -> bool {
                let elem_size = get_dtype_info(self.dtype).unwrap().size;
                let expected = Tensor::compute_row_major_strides(&self.shape, elem_size);
                self.strides == expected
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
                self.dtype
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

            // 其他宏展开
            $crate::impl_device_transfer!($name, $lock, $into_handle);
            $crate::impl_broadcast_to!($name, $lock, $into_handle);
            $crate::impl_transpose!($name, $lock, $into_handle);
            $crate::impl_slice!($name, $lock, $into_handle);
            $crate::impl_concat_split!($name, $lock, $into_handle);
            $crate::impl_strided_copy_to!($name, $lock, $into_handle);
            $crate::impl_contiguous!($name, $lock, $into_handle);
            $crate::impl_matmul_with_out!($name, $lock, $into_handle);
            $crate::impl_matmul!($name, $lock, $into_handle);
        }

        // 加法操作宏也需要修改
        $crate::impl_add_for_view!($name, $lock, $into_handle);
    };
}
impl_tensor_view!(RcTensorView, RcTensor, lock_rc, into_rc);
impl_tensor_view!(ArcTensorView, ArcTensor, lock_arc, into_arc);

pub trait AsView {
    type View: TensorViewOps;
    fn as_view(&self) -> Self::View;
}

macro_rules! define_view_to_vec {
    ($func_name:ident, $view_type:ident, $into_handle:expr, $lock:ident) => {
        fn $func_name<T: bytemuck::Pod + crate::dtype::DTypeMapping>(view: &$view_type) -> Vec<T> {
            let cpu_view = view.to_cpu().expect("Failed to copy to CPU");
            let cell = $lock(&cpu_view.handle);
            let tensor = cell.borrow(); // 添加 .borrow()
            let bytes = tensor.as_bytes().expect("Failed to get bytes");
            let result = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const T, view.size()).to_vec()
            };
            result
        }
    };
}

define_view_to_vec!(rc_view_to_vec, RcTensorView, into_rc, lock_rc);
define_view_to_vec!(arc_view_to_vec, ArcTensorView, into_arc, lock_arc);

pub fn rc_view_to_vec_f32(view: &RcTensorView) -> Vec<f32> {
    rc_view_to_vec(view)
}
pub fn arc_view_to_vec_f32(view: &ArcTensorView) -> Vec<f32> {
    arc_view_to_vec(view)
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
    fn test_stream_wait_event() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0).unwrap();

        let stream1 = cuda::Stream::new(None).unwrap(); // 使用 cuda::Stream
        let stream2 = cuda::Stream::new(None).unwrap();

        // 创建 CPU 数据并上传到 GPU
        let a_cpu = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b_cpu = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);

        let a_gpu = a_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let b_gpu = b_cpu.into_arc().as_view().to_gpu(0).unwrap();

        // 创建输出张量（全零） - 需要两个独立的零张量，因为 into_arc 消耗所有权
        let zero_cpu1 = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32).unwrap();
        let mut out_gpu = zero_cpu1.into_arc().as_view().to_gpu(0).unwrap();

        // 异步加法（使用默认流）
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        let event = stream1.record().unwrap();
        stream2.wait_event(&event).unwrap();

        // 第二个输出张量
        let zero_cpu2 = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32).unwrap();
        let mut out2_gpu = zero_cpu2.into_arc().as_view().to_gpu(0).unwrap();
        ArcTensorView::add(&out_gpu, &out_gpu, &mut out2_gpu).unwrap();

        stream2.synchronize().unwrap();
        let result = arc_view_to_vec_f32(&out2_gpu);
        assert_eq!(result, vec![8.0, 12.0]);
    }

    #[test]
    fn test_device_context_switch() {
        if !cuda::is_available() {
            return;
        }
        let dev_count = get_cuda_device_count().unwrap();
        if dev_count < 2 {
            return;
        }

        cuda::set_device(0).unwrap();
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let a_view = a.into_arc().as_view();
        let a_gpu = a_view.to_gpu(0).unwrap();

        cuda::set_device(1);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let b_view = b.into_arc().as_view();
        let b_gpu = b_view.to_gpu(1).unwrap();

        // 不能直接跨设备加法，应该报错
        let zero_cpu = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32).unwrap();
        let mut out_gpu = zero_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let result = ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu);
        assert!(result.is_err());
    }
}
