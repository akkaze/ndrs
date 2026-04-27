/// 具体视图类型：RcTensorView 和 ArcTensorView
use super::slice::{SliceArg, SliceInfo};
use super::trait_def::TensorViewOps;
use crate::Device;
use crate::cuda;
use crate::cuda::Stream;
use crate::dtype::{DType, get_dtype_info};
use crate::kernel::*;
use crate::tensor::{ArcTensor, DataPtr, RcTensor, Tensor};
use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr};
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

// 修改宏定义，去掉 $lock 和 $into_handle 参数
macro_rules! impl_tensor_view {
    ($name:ident, $handle:ty) => {
        // 只保留类型名和句柄类型
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
                    let cell = handle.lock(); // 直接使用 handle.lock()
                    let tensor = cell.borrow();
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

            fn create_output(&self) -> anyhow::Result<Self> {
                self.create_output_on_device(self.device)
            }

            fn create_output_on_device(&self, device: Device) -> anyhow::Result<Self> {
                let elem_size = get_dtype_info(self.dtype).unwrap().size;
                let total_bytes = self.size() * elem_size;
                let shape = self.shape().to_vec();
                let dtype = self.dtype;

                let new_tensor = match device {
                    Device::Cpu => {
                        let bytes = vec![0u8; total_bytes].into_boxed_slice();
                        Tensor::new_cpu_from_bytes(bytes, shape, dtype)
                            .context("Failed to create CPU tensor")?
                    }
                    Device::Cuda(dev_id) => {
                        let stream =
                            $crate::cuda::get_stream().context("Failed to get CUDA stream")?;
                        if stream.device_id != dev_id {
                            bail!(
                                "Stream device {} does not match target device {}",
                                stream.device_id,
                                dev_id
                            );
                        }
                        let gpu_mem = stream
                            .inner()
                            .alloc_zeros::<u8>(total_bytes)
                            .context("Failed to allocate GPU memory")?;
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
                Ok(Self::new(<$handle>::from_tensor(new_tensor)))
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

            fn device(&self) -> Device {
                self.device
            }

            fn assign(&mut self, src: &Self) -> anyhow::Result<()> {
                if self.shape != src.shape {
                    bail!("Shape mismatch");
                }
                src.strided_copy_to(self)
            }

            /// 获取 GPU 切片（只读），视图偏移量由内核参数 strides 和 offset 处理。
            #[cfg(feature = "cuda")]
            unsafe fn as_gpu_slice(&self) -> &CudaSlice<u8> {
                let cell = self.handle.lock();
                let tensor = cell.borrow();
                let slice = tensor.as_gpu_slice().expect("Not on GPU");
                // 安全：slice 的生命周期被延长，因为 cell 在函数结束时被丢，
                // 但 slice 实际指向的 GPU 内存由 handle 持有，不会失效。
                // 调用者必须保证在 slice 使用期间 handle 未被释放。
                std::mem::transmute(slice)
            }

            /// 获取 GPU 切片（可变）。
            #[cfg(feature = "cuda")]
            unsafe fn as_gpu_slice_mut(&mut self) -> &mut CudaSlice<u8> {
                let cell = self.handle.lock();
                let mut tensor = cell.borrow_mut();
                let slice = tensor.as_gpu_slice_mut().expect("Not on GPU");
                std::mem::transmute(slice)
            }

            #[cfg(feature = "cuda")]
            unsafe fn as_gpu_view(&self) -> CudaView<'_, u8> {
                let cell = self.handle.lock();
                let tensor = cell.borrow();
                let base_slice = tensor.as_gpu_slice().expect("Not on GPU");
                let offset = self.offset(); // 已是字节偏移
                let len = self.num_bytes(); // 使用 num_bytes 获取实际字节数
                let view = base_slice.slice(offset..);
                std::mem::transmute(view)
            }

            #[cfg(feature = "cuda")]
            unsafe fn as_gpu_view_mut(&self) -> CudaViewMut<'_, u8> {
                let cell = self.handle.lock();
                let mut tensor = cell.borrow_mut();
                let base_slice = tensor.as_gpu_slice_mut().expect("Not on GPU");
                let offset = self.offset();
                let len = self.num_bytes();
                let view = base_slice.slice_mut(offset..);
                std::mem::transmute(view)
            }

            unsafe fn raw_data_ptr(&self) -> *mut u8 {
                let cell = self.handle.lock();
                let tensor = cell.borrow();
                let base_ptr = tensor.data_ptr(None);
                (base_ptr as *mut u8).add(self.offset())
            }

            // 直接调用不带参数的辅助宏，这些宏内部使用 handle.lock() 和 <$handle>::from_tensor
            $crate::impl_device_transfer!($name, $handle);
            $crate::impl_broadcast_to!($name, $handle);
            $crate::impl_transpose!($name, $handle);
            $crate::impl_slice!($name, $handle);
            $crate::impl_concat_split!($name, $handle);
            $crate::impl_strided_copy_to!($name, $handle);
            $crate::impl_contiguous!($name, $handle);
            $crate::impl_matmul_into!($name, $handle);
            $crate::impl_matmul!($name, $handle);
            $crate::impl_fill!($name, $handle);
        }

        // 加法宏也不再需要额外参数
        $crate::impl_add_for_view!($name, $handle);
    };
}

// 调用宏，只传递类型名称和句柄类型
impl_tensor_view!(RcTensorView, RcTensor);
impl_tensor_view!(ArcTensorView, ArcTensor);

// define_view_to_vec 宏也做相应修改，去掉 lock 辅助函数
macro_rules! define_view_to_vec {
    ($func_name:ident, $view_type:ident) => {
        fn $func_name<T: bytemuck::Pod + crate::dtype::DTypeMapping>(view: &$view_type) -> Vec<T> {
            let cpu_view = view.to_cpu().expect("Failed to copy to CPU");
            let cell = cpu_view.handle.lock();
            let tensor = cell.borrow();
            let bytes = tensor.as_bytes().expect("Failed to get bytes");
            let result = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const T, view.size()).to_vec()
            };
            result
        }
    };
}

define_view_to_vec!(rc_view_to_vec, RcTensorView);
define_view_to_vec!(arc_view_to_vec, ArcTensorView);

pub fn rc_view_to_vec_f32(view: &RcTensorView) -> Vec<f32> {
    rc_view_to_vec(view)
}
pub fn arc_view_to_vec_f32(view: &ArcTensorView) -> Vec<f32> {
    arc_view_to_vec(view)
}

impl RcTensorView {
    pub fn lock(&self) -> &RefCell<Tensor> {
        &self.handle.0
    }
}

impl ArcTensorView {
    pub fn lock(&self) -> parking_lot::ReentrantMutexGuard<RefCell<Tensor>> {
        self.handle.0.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DTYPE_FLOAT32;
    use crate::cuda::{
        self, get_device_count as get_cuda_device_count, is_available as cuda_available,
        set_device as set_current_device,
    };
    use crate::s;
    use crate::tensor::Tensor;
    use crate::view::trait_def::TensorViewOps;

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
        transposed.contiguous_into(&mut out_view).unwrap();
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
        transposed.contiguous_into(&mut out_view).unwrap();
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
        let zero_cpu1 = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32, Device::Cpu).unwrap();
        let mut out_gpu = zero_cpu1.into_arc().as_view().to_gpu(0).unwrap();

        // 异步加法（使用默认流）
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        let event = stream1.record().unwrap();
        stream2.wait_event(&event).unwrap();

        // 第二个输出张量
        let zero_cpu2 = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32, Device::Cpu).unwrap();
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
        let zero_cpu = Tensor::new_contiguous(vec![2], DTYPE_FLOAT32, Device::Cpu).unwrap();
        let mut out_gpu = zero_cpu.into_arc().as_view().to_gpu(0).unwrap();
        let result = ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_event_timing_and_wait() {
        if !cuda::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }
        cuda::set_device(0).unwrap();

        let stream1 = cuda::Stream::new(Some(0)).unwrap();
        let stream2 = cuda::Stream::new(Some(0)).unwrap();

        let size = 1024 * 1024;
        let shape = vec![1024, 1024];

        // 在 stream1 上执行加法
        cuda::set_stream(stream1.clone()).unwrap();
        let a = Tensor::new_cpu_from_f32(vec![1.0; size], shape.clone());
        let b = Tensor::new_cpu_from_f32(vec![2.0; size], shape.clone());
        let a_gpu = a.into_arc().as_view().to_gpu(0).unwrap();
        let b_gpu = b.into_arc().as_view().to_gpu(0).unwrap();
        let mut out1 = Tensor::new_contiguous(shape.clone(), DTYPE_FLOAT32, Device::Cpu)
            .unwrap()
            .into_arc()
            .as_view()
            .to_gpu(0)
            .unwrap();
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out1).unwrap();

        // 在 stream1 上记录事件
        let event = stream1.record().unwrap();

        // 切换到 stream2，等待事件后执行加法
        cuda::set_stream(stream2.clone()).unwrap();
        stream2.wait_event(&event).unwrap();

        let mut out2 = Tensor::new_contiguous(shape, DTYPE_FLOAT32, Device::Cpu)
            .unwrap()
            .into_arc()
            .as_view()
            .to_gpu(0)
            .unwrap();
        ArcTensorView::add(&out1, &out1, &mut out2).unwrap();

        stream2.synchronize().unwrap();

        let result = arc_view_to_vec_f32(&out2);
        let expected: Vec<f32> = vec![6.0; size];
        assert_eq!(result, expected);
    }
    #[test]
    fn test_event_elapsed_custom_stream() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0).unwrap();
        let stream = cuda::Stream::new(Some(0)).unwrap();
        cuda::set_stream(stream.clone()).unwrap();

        let a = Tensor::new_cpu_from_f32(vec![1.0; 1024 * 1024], vec![1024, 1024])
            .into_arc()
            .as_view()
            .to_gpu(0)
            .unwrap();
        let b = Tensor::new_cpu_from_f32(vec![2.0; 1024 * 1024], vec![1024, 1024])
            .into_arc()
            .as_view()
            .to_gpu(0)
            .unwrap();
        let mut out = Tensor::new_cpu_from_f32(vec![0.0; 1024 * 1024], vec![1024, 1024])
            .into_arc()
            .as_view()
            .to_gpu(0)
            .unwrap();

        let start = stream.record().unwrap();
        ArcTensorView::add(&a, &b, &mut out).unwrap();
        let end = stream.record().unwrap();
        stream.synchronize().unwrap();

        let elapsed = end.elapsed_since(&start).unwrap();
        println!("Elapsed: {:?}", elapsed);
    }
}
