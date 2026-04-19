use crate::device::Device;
use crate::dtype::{get_add_op, get_dtype_info, DType};
use crate::kernel::*;
use crate::tensor::{DataPtr, Tensor};
use cudarc::driver::DevicePtr;
use std::cell::RefCell;
use std::ops::{Add, AddAssign};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// ---------- 切片类型 ----------
#[derive(Debug, Clone)]
pub enum SliceArg {
    Index(usize),
    Range(usize, usize, usize), // start, end, step
    All,
}

/// 切片信息，由 s! 宏生成
pub struct SliceInfo {
    args: Vec<SliceArg>,
}

impl SliceInfo {
    pub fn new(args: Vec<SliceArg>) -> Self {
        SliceInfo { args }
    }
    pub fn args(&self) -> &[SliceArg] {
        &self.args
    }
}

// ---------- 公共 trait ----------
pub trait TensorViewOps: Clone {
    type Handle: Clone;
    fn new(handle: Self::Handle) -> Self;
    fn as_strided(&self, new_shape: Vec<usize>, new_strides: Vec<usize>, offset: usize) -> Self;
    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String>;
    fn transpose(&self, axes: &[usize]) -> Result<Self, String>;
    fn T(&self) -> Result<Self, String> {
        let mut axes: Vec<usize> = (0..self.shape().len()).rev().collect();
        self.transpose(&axes)
    }
    fn strided_copy_to(&self, dst: &mut Self) -> Result<(), String>;
    fn contiguous(&self, out: &mut Self) -> Result<(), String>;
    fn to(&self, out: &mut Self, target_device: Device) -> Result<(), String>;
    fn to_cpu(&self, out: &mut Self) -> Result<(), String> {
        self.to(out, Device::CPU)
    }
    fn to_gpu(&self, out: &mut Self, device_id: usize) -> Result<(), String> {
        self.to(out, Device::GPU(device_id))
    }
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn dtype(&self) -> DType;
    fn size(&self) -> usize;
    fn assign(&mut self, src: &Self) -> Result<(), String>;
    fn slice(&self, info: &SliceInfo) -> Result<Self, String>;
}

// ---------- 辅助宏：为两种视图实现相同逻辑 ----------
macro_rules! impl_tensor_view {
    ($name:ident, $handle:ty, $borrow:ident, $borrow_mut:ident, $lock:expr, $into_handle:expr) => {
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

            // 辅助：创建相同形状的空输出张量（连续，CPU）
            fn create_output(&self) -> Result<Self, String> {
                let elem_size = get_dtype_info(self.dtype()).unwrap().size;
                let total_bytes = self.size() * elem_size;
                let tensor = crate::tensor::Tensor::new_cpu_from_bytes(
                    vec![0u8; total_bytes].into_boxed_slice(),
                    self.shape().to_vec(),
                    self.dtype(),
                )?;
                Ok($name::new($into_handle(tensor)))
            }

            pub fn into_handle(self) -> $handle {
                self.handle
            }
            pub fn handle(&self) -> &$handle {
                &self.handle
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
            fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
                if self.shape.len() > target_shape.len() {
                    return Err("Cannot broadcast to shape with fewer dimensions".into());
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
                        return Err(format!(
                            "Cannot broadcast dim {}: {} vs {}",
                            i, self_dim, target_dim
                        ));
                    }
                }
                for i in 0..offset {
                    new_strides[i] = 0;
                }
                Ok(self.as_strided(target_shape.to_vec(), new_strides, 0))
            }
            fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
                if axes.len() != self.shape.len() {
                    return Err(format!(
                        "Number of axes ({}) does not match tensor dimensions ({})",
                        axes.len(),
                        self.shape.len()
                    ));
                }
                let mut new_shape = Vec::with_capacity(self.shape.len());
                let mut new_strides = Vec::with_capacity(self.shape.len());
                let mut used = vec![false; self.shape.len()];
                for &axis in axes {
                    if axis >= self.shape.len() {
                        return Err(format!(
                            "Axis {} out of range (ndim={})",
                            axis,
                            self.shape.len()
                        ));
                    }
                    if used[axis] {
                        return Err(format!("Axis {} repeated in permutation", axis));
                    }
                    used[axis] = true;
                    new_shape.push(self.shape[axis]);
                    new_strides.push(self.strides[axis]);
                }
                Ok(self.as_strided(new_shape, new_strides, self.offset))
            }
            fn strided_copy_to(&self, dst: &mut Self) -> Result<(), String> {
                if self.shape != dst.shape {
                    return Err(format!(
                        "Shape mismatch in strided_copy_to: self shape {:?}, dst shape {:?}",
                        self.shape, dst.shape
                    ));
                }
                let src_tensor = $borrow(&self.handle);
                let mut dst_tensor = $borrow_mut(&dst.handle);
                if src_tensor.dtype() != dst_tensor.dtype() {
                    return Err(format!(
                        "Dtype mismatch in strided_copy_to: self dtype {:?}, dst dtype {:?}",
                        src_tensor.dtype(),
                        dst_tensor.dtype()
                    ));
                }
                if src_tensor.device() != dst_tensor.device() {
                    return Err(format!(
                        "Device mismatch in strided_copy_to: self device {:?}, dst device {:?}",
                        src_tensor.device(),
                        dst_tensor.device()
                    ));
                }
                let total_elements: usize = self.shape.iter().product();
                let elem_size = get_dtype_info(src_tensor.dtype()).unwrap().size;
                match src_tensor.device() {
                    Device::CPU => {
                        let src_bytes = src_tensor.data_ptr(None);
                        let dst_bytes = dst_tensor.data_mut_ptr(None);
                        unsafe {
                            cpu_strided_copy(
                                src_bytes,
                                self.offset,
                                self.strides.as_ptr(),
                                self.shape.len() as i32,
                                self.shape.as_ptr(),
                                dst_bytes,
                                dst.offset,
                                dst.strides.as_ptr(),
                                elem_size,
                                total_elements,
                            );
                        }
                    }
                    Device::GPU(_) => {
                        let ctx = src_tensor.cuda_ctx_ref().unwrap();
                        let stream = &ctx.stream;
                        let shape_dev =
                            stream.clone_htod(&self.shape).map_err(|e| e.to_string())?;
                        let src_strides_dev = stream
                            .clone_htod(&self.strides)
                            .map_err(|e| e.to_string())?;
                        let dst_strides_dev =
                            stream.clone_htod(&dst.strides).map_err(|e| e.to_string())?;
                        let src_bytes = src_tensor.data_ptr(Some(stream));
                        let dst_bytes = dst_tensor.data_mut_ptr(Some(stream));
                        let stream_ptr = ctx.stream_ptr();
                        let (src_strides_ptr, _) = src_strides_dev.device_ptr(stream);
                        let (shape_ptr, _) = shape_dev.device_ptr(stream);
                        let (dst_strides_ptr, _) = dst_strides_dev.device_ptr(stream);
                        unsafe {
                            gpu_strided_copy(
                                src_bytes,
                                self.offset,
                                src_strides_ptr as *const usize,
                                self.shape.len() as i32,
                                shape_ptr as *const usize,
                                dst_bytes,
                                dst.offset,
                                dst_strides_ptr as *const usize,
                                elem_size,
                                total_elements,
                                stream_ptr,
                            );
                        }
                        stream.synchronize().map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            }
            fn contiguous(&self, out: &mut Self) -> Result<(), String> {
                if out.shape != self.shape {
                    return Err("Output shape mismatch for contiguous".into());
                }
                if !$borrow(&out.handle).is_contiguous() {
                    return Err("Output must be contiguous".into());
                }
                self.strided_copy_to(out)
            }
            fn to(&self, out: &mut Self, target_device: Device) -> Result<(), String> {
                eprintln!("[to] entered, target_device: {:?}", target_device);
                if self.shape != out.shape {
                    return Err("Shape mismatch in to".into());
                }

                // 检测是否同一句柄（死锁预防）
                if std::ptr::eq(&*self.handle, &*out.handle) {
                    eprintln!(
                        "[to] self and out are the same handle, performing copy via temporary"
                    );
                    // 创建一个临时副本（深拷贝数据），然后转换设备
                    let mut temp = self.create_output()?;
                    self.strided_copy_to(&mut temp)?;
                    return temp.to(out, target_device); // 递归调用，但此时句柄不同
                }

                // 获取源张量的不可变锁
                let src_t = $borrow(&self.handle);
                eprintln!("[to] src_t locked, device = {:?}", src_t.device());
                // 获取目标张量的不可变锁（仅用于检查 dtype）
                let dst_t = $borrow(&out.handle);
                if src_t.dtype() != dst_t.dtype() {
                    return Err("Dtype mismatch".into());
                }
                drop(dst_t); // 释放目标张量的锁，后续需要可变锁时再获取
                match (src_t.device(), target_device) {
                    (a, b) if a == b => {
                        eprintln!(
                            "[to] same device, releasing src_t lock and calling strided_copy_to"
                        );
                        drop(src_t); // 释放源锁，避免 strided_copy_to 中再次锁定导致死锁
                        self.strided_copy_to(out)
                    }
                    (Device::CPU, Device::GPU(idx)) => {
                        eprintln!("[to] CPU->GPU branch entered, idx={}", idx);
                        let mut dst_t = $borrow_mut(&out.handle);
                        eprintln!("[to] dst_t locked");
                        let ctx = crate::device::get_or_create_context(idx)?;
                        eprintln!("[to] ctx obtained");
                        let bytes = match &src_t.data {
                            crate::tensor::DataPtr::Cpu(b) => {
                                eprintln!("[to] src_t.data is CPU, len={}", b.len());
                                b.as_ref()
                            }
                            _ => unreachable!(),
                        };
                        eprintln!("[to] before clone_htod");
                        let gpu_mem = ctx
                            .stream
                            .clone_htod::<u8, _>(bytes)
                            .map_err(|e| e.to_string())?;
                        eprintln!("[to] clone_htod succeeded");
                        dst_t.data = crate::tensor::DataPtr::Gpu(gpu_mem);
                        dst_t.device = Device::GPU(idx);
                        dst_t.cuda_ctx = Some(ctx);
                        eprintln!("[to] CPU->GPU transfer complete");
                        Ok(())
                    }
                    (Device::GPU(_), Device::CPU) => {
                        eprintln!("[to] GPU -> CPU");
                        let mut dst_t = $borrow_mut(&out.handle);
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
                        dst_t.data = crate::tensor::DataPtr::Cpu(bytes.into_boxed_slice().into());
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
                    return Err("Shape mismatch in assign".into());
                }
                src.strided_copy_to(self)
            }
            fn slice(&self, info: &SliceInfo) -> Result<Self, String> {
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
                        SliceArg::Index(idx) => {
                            if *idx >= dim_size {
                                return Err("Index out of bounds".into());
                            }
                            new_offset += idx * dim_stride;
                            // 降维，不添加新维度
                        }
                        SliceArg::Range(start, end, step) => {
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
                        SliceArg::All => {
                            new_shape.push(dim_size);
                            new_strides.push(dim_stride);
                        }
                    }
                }
                // 未切片的维度保持原样
                for dim in slices.len()..self.shape.len() {
                    new_shape.push(self.shape[dim]);
                    new_strides.push(self.strides[dim]);
                }
                Ok(self.as_strided(new_shape, new_strides, new_offset))
            }
        }

        // ---------- 运算符重载 ----------
        impl Add for $name {
            type Output = Self;
            fn add(self, other: Self) -> Self::Output {
                assert_eq!(self.shape(), other.shape(), "Shapes must match for Add");
                let mut out = self.create_output().expect("Failed to create output");
                Self::add(&self, &other, &mut out).expect("Addition failed");
                out
            }
        }

        impl AddAssign for $name {
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
                Self::add(&temp, &other, &mut result).expect("Addition failed");
                result.strided_copy_to(self).expect("Copy back failed");
            }
        }

        impl $name {
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
                if a_t.dtype() != b_t.dtype() || a_t.dtype() != out_t.dtype() {
                    return Err("Dtype mismatch in add".into());
                }
                let n = a.shape.iter().product();
                let add_op = get_add_op(a_t.dtype()).expect("Add op not registered");
                let stream_opt = match a_t.device() {
                    Device::CPU => None,
                    Device::GPU(_) => Some(&a_t.cuda_ctx_ref().unwrap().stream),
                };
                let a_ptr = a_t.data_ptr(stream_opt);
                let b_ptr = b_t.data_ptr(stream_opt);
                let c_ptr = out_t.data_mut_ptr(stream_opt);
                let c_stream = match a_t.device() {
                    Device::CPU => None,
                    Device::GPU(_) => Some(a_t.cuda_ctx_ref().unwrap().stream_ptr()),
                };
                add_op(a_ptr, b_ptr, c_ptr, n, a_t.device(), c_stream);
                if let Device::GPU(_) = a_t.device() {
                    a_t.cuda_ctx_ref()
                        .unwrap()
                        .stream
                        .synchronize()
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
        }
    };
}

// 定义锁函数
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

// 生成两个视图类型
impl_tensor_view!(
    RcTensorView,
    Rc<RefCell<Tensor>>,
    lock_rc,
    lock_rc_mut,
    lock_rc,
    into_rc
);
impl_tensor_view!(
    ArcTensorView,
    Arc<Mutex<Tensor>>,
    lock_arc,
    lock_arc_mut,
    lock_arc,
    into_arc
);

// 广播形状辅助函数
pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result = Vec::new();
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result.push(std::cmp::max(dim1, dim2));
        } else {
            return None;
        }
    }
    result.reverse();
    Some(result)
}

/// 扩展 trait，为智能指针提供 `as_view` 方法
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{self, cuda_available, get_cuda_device_count, set_current_device};
    use crate::s;
    use crate::tensor::Tensor;
    use crate::DTYPE_FLOAT32;

    // 辅助函数：将 RcTensorView 转换为连续 CPU 数据并返回 Vec<f32>
    fn rc_view_to_vec(view: &RcTensorView) -> Vec<f32> {
        let elem_size = std::mem::size_of::<f32>();
        let total_bytes = view.size() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            view.shape().to_vec(),
            view.dtype(),
        )
        .unwrap();
        let out_handle = out_tensor.into_rc();
        let mut out_view = RcTensorView::new(out_handle);
        view.contiguous(&mut out_view).unwrap();
        let tensor = out_view.handle.borrow();
        let bytes = tensor.as_bytes().unwrap();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, view.size()).to_vec() }
    }

    // 辅助函数：将 ArcTensorView 转换为连续 CPU 数据并返回 Vec<f32>
    fn arc_view_to_vec(view: &ArcTensorView) -> Vec<f32> {
        let elem_size = std::mem::size_of::<f32>();
        let total_bytes = view.size() * elem_size;
        // 创建全新的 CPU 张量（独立内存）
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            view.shape().to_vec(),
            view.dtype(),
        )
        .unwrap();
        let out_handle = out_tensor.into_arc(); // 新句柄
        let mut out_view = ArcTensorView::new(out_handle); // 新视图，与 view 的句柄不同
                                                           // 使用 to_cpu 传输数据
        view.to_cpu(&mut out_view).unwrap();
        let tensor = out_view.handle().lock().unwrap();
        let bytes = tensor.as_bytes().unwrap();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, view.size()).to_vec() }
    }

    // ---------- RcTensorView 测试 ----------
    #[test]
    fn test_rc_view_creation() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = t.into_rc().as_view();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[8, 4]);
        assert_eq!(view.offset(), 0);
    }

    #[test]
    fn test_rc_slice() {
        let t = Tensor::new_cpu_from_f32((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let view = t.into_rc().as_view();
        let sub = view.slice(&s![1..3, 2..4]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        let expected = vec![6.0, 7.0, 10.0, 11.0];
        assert_eq!(rc_view_to_vec(&sub), expected);
    }

    #[test]
    fn test_rc_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        a_view.assign(&b_view).unwrap();
        assert_eq!(rc_view_to_vec(&a_view), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_rc_add() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let c_view = a_view + b_view;
        assert_eq!(rc_view_to_vec(&c_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_rc_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        a_view += b_view;
        assert_eq!(rc_view_to_vec(&a_view), vec![4.0, 6.0]);
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
            rc_view_to_vec(&out_view),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
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
    fn test_rc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_rc().as_view();
        let mut dst_view = dst.into_rc().as_view();
        src_view.to_cpu(&mut dst_view).unwrap();
        assert_eq!(rc_view_to_vec(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_rc_slice_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![10.0, 20.0], vec![1, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let mut sub = a_view.slice(&s![0..1, ..]).unwrap();
        sub.assign(&b_view).unwrap();
        assert_eq!(rc_view_to_vec(&a_view), vec![10.0, 20.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rc_slice_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0], vec![1, 2]);
        let mut a_view = a.into_rc().as_view();
        let b_view = b.into_rc().as_view();
        let mut sub = a_view.slice(&s![1..2, ..]).unwrap();
        sub += b_view;
        assert_eq!(rc_view_to_vec(&a_view), vec![1.0, 2.0, 8.0, 10.0]);
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
    fn test_arc_slice() {
        let t = Tensor::new_cpu_from_f32((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let view = t.into_arc().as_view();
        let sub = view.slice(&s![1..3, 2..4]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        let expected = vec![6.0, 7.0, 10.0, 11.0];
        assert_eq!(arc_view_to_vec(&sub), expected);
    }

    #[test]
    fn test_arc_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        a_view.assign(&b_view).unwrap();
        assert_eq!(arc_view_to_vec(&a_view), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_arc_add() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let c_view = a_view + b_view;
        assert_eq!(arc_view_to_vec(&c_view), vec![4.0, 6.0]);
    }

    #[test]
    fn test_arc_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new_cpu_from_f32(vec![3.0, 4.0], vec![2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        a_view += b_view;
        assert_eq!(arc_view_to_vec(&a_view), vec![4.0, 6.0]);
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
            arc_view_to_vec(&out_view),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
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
    fn test_arc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.to_cpu(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_slice_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![10.0, 20.0], vec![1, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let mut sub = a_view.slice(&s![0..1, ..]).unwrap();
        sub.assign(&b_view).unwrap();
        assert_eq!(arc_view_to_vec(&a_view), vec![10.0, 20.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arc_slice_add_assign() {
        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new_cpu_from_f32(vec![5.0, 6.0], vec![1, 2]);
        let mut a_view = a.into_arc().as_view();
        let b_view = b.into_arc().as_view();
        let mut sub = a_view.slice(&s![1..2, ..]).unwrap();
        sub += b_view;
        assert_eq!(arc_view_to_vec(&a_view), vec![1.0, 2.0, 8.0, 10.0]);
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
        // 步长应为 [4, 12]（原始连续步长 [12, 4] 交换）
        assert_eq!(transposed.strides(), &[4, 12]);
        // 验证数据
        let expected = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        assert_eq!(rc_view_to_vec(&transposed), expected);
    }

    #[test]
    fn test_T() {
        let t = Tensor::new_cpu_from_f32((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let view = t.into_rc().as_view();
        let transposed = view.T().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[4, 12]);
        assert_eq!(
            rc_view_to_vec(&transposed),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_transpose_3d() {
        let t = Tensor::new_cpu_from_f32((0..24).map(|x| x as f32).collect(), vec![2, 3, 4]);
        let view = t.into_rc().as_view();
        let transposed = view.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape(), &[4, 2, 3]);
        // 原始连续步长: [48, 16, 4] (2*3*4=24个元素，每个4字节，行步长48，列步长16，深度步长4)
        // 转置后步长: [4, 48, 16]
        assert_eq!(transposed.strides(), &[4, 48, 16]);
    }

    #[test]
    fn test_arc_same_device_copy() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.strided_copy_to(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_strided_copy_to() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        src_view.strided_copy_to(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec(&dst_view), vec![1.0, 2.0]);
    }

    // ---------- GPU 测试（需要 CUDA 设备）----------
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

        // 传输输入到 GPU
        let mut a_gpu = a_view.clone();
        let mut b_gpu = b_view.clone();
        a_view.to_gpu(&mut a_gpu, 0).unwrap();
        b_view.to_gpu(&mut b_gpu, 0).unwrap();

        // 创建独立输出 GPU 张量
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

        // 执行加法
        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();

        // 使用 arc_view_to_vec 自动处理拷贝到 CPU
        let result = arc_view_to_vec(&out_gpu);
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_gpu_add_with_stream() {
        use crate::stream::Stream;
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream = Stream::new().unwrap();

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

        // 在自定义流中执行加法（注意：当前 add 函数使用默认流，若需使用自定义流需修改 add 支持 stream 参数）
        // 这里仅演示流同步：先记录事件，然后等待
        let event = stream.record().unwrap();
        stream.wait_event(&event).unwrap(); // 等待自己无意义，但展示用法

        ArcTensorView::add(&a_gpu, &b_gpu, &mut out_gpu).unwrap();
        stream.synchronize().unwrap();

        let result = arc_view_to_vec(&out_gpu);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }
    #[test]
    fn test_gpu_to_cpu_transfer() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let src_tensor = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let src_view = src_tensor.into_arc().as_view();

        // 创建独立的 GPU 输出张量
        let elem_size = std::mem::size_of::<f32>();
        let total_bytes = src_view.size() * elem_size;
        let out_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            src_view.shape().to_vec(),
            src_view.dtype(),
        )
        .unwrap();
        let mut gpu_view = out_tensor.into_arc().as_view();

        src_view.to_gpu(&mut gpu_view, 0).unwrap(); // 不同句柄
        assert_eq!(gpu_view.handle().lock().unwrap().device(), Device::GPU(0));

        // 创建独立的 CPU 输出张量
        let back_tensor = Tensor::new_cpu_from_bytes(
            vec![0u8; total_bytes].into_boxed_slice(),
            src_view.shape().to_vec(),
            src_view.dtype(),
        )
        .unwrap();
        let mut back_cpu = back_tensor.into_arc().as_view();
        gpu_view.to_cpu(&mut back_cpu).unwrap();
        assert_eq!(arc_view_to_vec(&back_cpu), vec![1.0, 2.0, 3.0]);
    }
}
