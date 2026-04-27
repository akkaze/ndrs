/// 设备间数据传输
use crate::dtype::get_dtype_info;
use crate::tensor::DataPtr;
use crate::view::TensorViewOps;
use anyhow::{Context, Result, anyhow, bail};

#[macro_export]
macro_rules! impl_device_transfer {
    ($view_type:ident, $handle:ty) => {
        fn to(&self, out: &mut Self, target_device: Device) -> anyhow::Result<()> {
            use anyhow::{Context, bail};

            if self.shape != out.shape {
                bail!("Shape mismatch");
            }

            match (self.device, target_device) {
                (src_dev, dst_dev) if src_dev == dst_dev => self.strided_copy_to(out),
                (Device::Cpu, Device::Cuda(idx)) => {
                    if !self.is_contiguous() {
                        let mut temp = self.create_output()?;
                        self.contiguous_into(&mut temp)?;
                        return temp.to(out, target_device);
                    }
                    let src_cell = self.handle.lock();
                    let src_tensor = src_cell.borrow();
                    let dst_cell = out.handle.lock();
                    let mut dst_tensor = dst_cell.borrow_mut();
                    let dst_gpu = match &mut dst_tensor.data {
                        DataPtr::Gpu(s) => s,
                        _ => bail!("Output is not GPU memory"),
                    };
                    let src_bytes = match &src_tensor.data {
                        DataPtr::Cpu(b) => b.as_ref(),
                        _ => unreachable!(),
                    };
                    let stream = $crate::cuda::get_stream().context("Failed to get CUDA stream")?;
                    stream
                        .inner()
                        .memcpy_htod(src_bytes, dst_gpu)
                        .context("Failed to copy to GPU")?;
                    stream.synchronize().context("Failed to synchronize")?;
                    dst_tensor.device = Device::Cuda(idx);
                    Ok(())
                }
                (Device::Cuda(_), Device::Cpu) => {
                    if !self.is_contiguous() {
                        let mut temp = self.create_output_on_device(Device::Cpu)?;
                        self.contiguous_into(&mut temp)?;
                        return temp.to(out, target_device);
                    }
                    let src_cell = self.handle.lock();
                    let src_tensor = src_cell.borrow();
                    let dst_cell = out.handle.lock();
                    let mut dst_tensor = dst_cell.borrow_mut();
                    let src_gpu = match &src_tensor.data {
                        DataPtr::Gpu(s) => s,
                        _ => bail!("Source is not GPU memory"),
                    };
                    let dst_bytes = match &mut dst_tensor.data {
                        DataPtr::Cpu(b) => b.as_mut(),
                        _ => bail!("Output is not CPU memory"),
                    };
                    let stream = $crate::cuda::get_stream().context("Failed to get CUDA stream")?;
                    stream
                        .inner()
                        .memcpy_dtoh(src_gpu, dst_bytes)
                        .context("Failed to copy to CPU")?;
                    stream.synchronize().context("Failed to synchronize")?;
                    dst_tensor.device = Device::Cpu;
                    Ok(())
                }
                (Device::Cuda(src_idx), Device::Cuda(dst_idx)) if src_idx != dst_idx => {
                    let mut cpu_temp = self.create_output()?;
                    self.to(&mut cpu_temp, Device::Cpu)?;
                    cpu_temp.to(out, target_device)
                }
                _ => bail!("Unsupported device conversion"),
            }
        }

        fn to_device(&self, target_device: Device) -> anyhow::Result<Self> {
            let mut out = self.create_output_on_device(target_device)?;
            self.to(&mut out, target_device)?;
            Ok(out)
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DTYPE_FLOAT32;
    use crate::Device;
    use crate::cuda::{
        self, get_device_count as get_cuda_device_count, is_available as cuda_available,
        set_device as set_current_device,
    };
    use crate::s;
    use crate::tensor::Tensor;
    use crate::view::{arc_view_to_vec_f32, rc_view_to_vec_f32};

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
