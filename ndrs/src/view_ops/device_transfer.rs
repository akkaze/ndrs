//! 设备间数据传输

use crate::dtype::get_dtype_info;
use crate::tensor::DataPtr;
use crate::view::TensorViewOps;

#[macro_export]
macro_rules! impl_device_transfer {
    ($view_type:ident, $lock:ident, $into_handle:expr) => {
        fn to(&self, out: &mut Self, target_device: Device) -> Result<(), String> {
            if self.shape != out.shape {
                return Err("Shape mismatch".into());
            }

            match (self.device, target_device) {
                (src_dev, dst_dev) if src_dev == dst_dev => {
                    // 设备相同，直接复制（strided_copy_to 内部会自行加锁）
                    self.strided_copy_to(out)
                }
                (Device::Cpu, Device::Cuda(idx)) => {
                    // 检查连续性，如果不连续则先连续化到临时 CPU 张量
                    if !self.is_contiguous() {
                        let mut temp = self.create_output()?;
                        self.contiguous(&mut temp)?;
                        return temp.to(out, target_device);
                    }
                    // 获取源和目标的锁
                    let src_cell = $lock(&self.handle);
                    let src_tensor = src_cell.borrow();
                    if self.dtype != src_tensor.dtype() {
                        return Err("Dtype mismatch".into());
                    }
                    let dst_cell = $lock(&out.handle);
                    let mut dst_tensor = dst_cell.borrow_mut();
                    if dst_tensor.dtype() != self.dtype {
                        return Err("Dtype mismatch".into());
                    }
                    let dst_gpu = match &mut dst_tensor.data {
                        DataPtr::Gpu(s) => s,
                        _ => return Err("Output is not GPU memory".into()),
                    };
                    let src_bytes = match &src_tensor.data {
                        DataPtr::Cpu(b) => b.as_ref(),
                        _ => unreachable!(),
                    };
                    let stream = cuda::get_stream().map_err(|e| e.to_string())?;
                    if let Err(e) = stream.inner().memcpy_htod(src_bytes, dst_gpu) {
                        return Err(e.to_string());
                    }
                    if let Err(e) = stream.synchronize() {
                        return Err(e.to_string());
                    }
                    dst_tensor.device = Device::Cuda(idx);
                    // 锁在此作用域结束后自动释放
                    Ok(())
                }
                (Device::Cuda(_), Device::Cpu) => {
                    if !self.is_contiguous() {
                        let mut temp = self.create_output_on_device(Device::Cpu)?;
                        self.contiguous(&mut temp)?;
                        return temp.to(out, target_device);
                    }
                    let src_cell = $lock(&self.handle);
                    let src_tensor = src_cell.borrow();
                    let dst_cell = $lock(&out.handle);
                    let mut dst_tensor = dst_cell.borrow_mut();
                    let src_gpu = match &src_tensor.data {
                        DataPtr::Gpu(s) => s,
                        _ => return Err("Source is not GPU memory".into()),
                    };
                    let dst_bytes = match &mut dst_tensor.data {
                        DataPtr::Cpu(b) => b.as_mut(),
                        _ => return Err("Output is not CPU memory".into()),
                    };
                    let stream = cuda::get_stream().map_err(|e| e.to_string())?;
                    if let Err(e) = stream.inner().memcpy_dtoh(src_gpu, dst_bytes) {
                        return Err(e.to_string());
                    }
                    if let Err(e) = stream.synchronize() {
                        return Err(e.to_string());
                    }
                    dst_tensor.device = Device::Cpu;
                    Ok(())
                }
                (Device::Cuda(src_idx), Device::Cuda(dst_idx)) if src_idx != dst_idx => {
                    // 不同 GPU 之间，通过 CPU 中转
                    let mut cpu_temp = self.create_output()?;
                    self.to(&mut cpu_temp, Device::Cpu)?;
                    cpu_temp.to(out, target_device)
                }
                _ => Err("Unsupported device conversion".into()),
            }
        }

        fn to_device(&self, target_device: Device) -> Result<Self, String> {
            let mut out = self.create_output_on_device(target_device)?;
            self.to(&mut out, target_device)?;
            Ok(out)
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
    use crate::Device;
    use crate::tensor::Tensor;
    use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};
    use crate::DTYPE_FLOAT32;

    #[test]
    fn test_rc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let src_view = src.into_rc().as_view();
        let dst_view = src_view.to_cpu().unwrap(); // 自动创建
        assert_eq!(rc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_arc_to_cpu() {
        let src = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let dst = Tensor::new_cpu_from_f32(vec![0.0, 0.0], vec![2]);
        let src_view = src.into_arc().as_view();
        let mut dst_view = dst.into_arc().as_view();
        // to_cpu 不再接受 &mut 参数，直接赋值
        let cpu_view = src_view.to_cpu().unwrap();
        // 手动复制到 dst_view
        cpu_view.strided_copy_to(&mut dst_view).unwrap();
        assert_eq!(arc_view_to_vec_f32(&dst_view), vec![1.0, 2.0]);
    }

    #[test]
    fn test_gpu_to_cpu_transfer() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0).unwrap();
        let src_tensor = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let src_view = src_tensor.into_arc().as_view();

        // 直接上传到 GPU，返回新视图
        let gpu_view = src_view.to_gpu(0).unwrap();
        assert_eq!(
            gpu_view.handle().0.lock().borrow().device(),
            Device::Cuda(0)
        );

        // 从 GPU 下载到 CPU
        let back_cpu = gpu_view.to_cpu().unwrap();
        assert_eq!(arc_view_to_vec_f32(&back_cpu), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_cpu_sync() {
        if !cuda::is_available() {
            return;
        }
        cuda::set_device(0).unwrap();

        let a = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let a_view = a.into_arc().as_view();
        let a_gpu = a_view.to_gpu(0).unwrap();

        // to_cpu 应该同步，返回后数据已就绪
        let cpu_out = a_gpu.to_cpu().unwrap();
        assert_eq!(arc_view_to_vec_f32(&cpu_out), vec![1.0, 2.0]);
    }
}
