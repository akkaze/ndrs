//! 设备间数据传输

use crate::view::TensorViewOps;
use crate::device::{Device, get_or_create_context};
use crate::dtype::get_dtype_info;
use crate::tensor::DataPtr;

pub fn to_impl<V: TensorViewOps>(src: &V, dst: &mut V, target_device: Device) -> Result<(), String> {
    if src.shape() != dst.shape() {
        return Err("Shape mismatch".into());
    }
    // 同一句柄死锁预防
    if std::ptr::eq(src.handle(), dst.handle()) {
        let mut temp = src.create_output()?;
        src.strided_copy_to(&mut temp)?;
        return to_impl(&temp, dst, target_device);
    }

    let src_t = src.handle().lock().unwrap();
    let mut dst_t = dst.handle().lock().unwrap();
    if src_t.dtype() != dst_t.dtype() {
        return Err("Dtype mismatch".into());
    }
    match (src_t.device(), target_device) {
        (a, b) if a == b => {
            drop(src_t);
            src.strided_copy_to(dst)
        }
        (Device::CPU, Device::GPU(idx)) => {
            let ctx = get_or_create_context(idx)?;
            let bytes = match &src_t.data {
                DataPtr::Cpu(b) => b.as_ref(),
                _ => unreachable!(),
            };
            let gpu_mem = ctx.stream.clone_htod::<u8, _>(bytes).map_err(|e| e.to_string())?;
            dst_t.data = DataPtr::Gpu(gpu_mem);
            dst_t.device = Device::GPU(idx);
            dst_t.cuda_ctx = Some(ctx);
            Ok(())
        }
        (Device::GPU(_), Device::CPU) => {
            let gpu_slice = match &src_t.data {
                DataPtr::Gpu(s) => s,
                _ => unreachable!(),
            };
            let bytes = src_t.cuda_ctx_ref()
                .ok_or("Missing CUDA context")?
                .stream.clone_dtoh(gpu_slice)
                .map_err(|e| e.to_string())?;
            dst_t.data = DataPtr::Cpu(bytes.into_boxed_slice());
            dst_t.device = Device::CPU;
            dst_t.cuda_ctx = None;
            Ok(())
        }
        _ => Err("Unsupported device conversion".into()),
    }
}