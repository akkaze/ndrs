use cudarc::driver::{CudaContext, CudaStream, DriverError};
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    CPU,
    GPU(usize),
}

thread_local! {
    static CURRENT_DEVICE: RefCell<Option<usize>> = RefCell::new(None);
}

pub fn set_current_device(dev_id: usize) {
    CURRENT_DEVICE.with(|c| *c.borrow_mut() = Some(dev_id));
}

pub fn get_current_device() -> Option<usize> {
    CURRENT_DEVICE.with(|c| *c.borrow())
}

#[derive(Clone)]
pub struct CudaContextWrapper {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub device_id: usize,
}

impl CudaContextWrapper {
    pub fn new(device_id: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(device_id)?;
        let stream = ctx.default_stream();
        Ok(CudaContextWrapper {
            ctx,
            stream,
            device_id,
        })
    }

    pub fn stream_ptr(&self) -> *mut std::ffi::c_void {
        self.stream.cu_stream() as *mut std::ffi::c_void
    }
}

static GPU_CONTEXTS: Lazy<Mutex<Vec<Option<Arc<CudaContextWrapper>>>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

pub fn get_or_create_context(device_id: usize) -> Result<Arc<CudaContextWrapper>, String> {
    let mut contexts = GPU_CONTEXTS.lock().unwrap();
    if device_id >= contexts.len() {
        contexts.resize(device_id + 1, None);
    }
    if let Some(ctx) = &contexts[device_id] {
        Ok(ctx.clone())
    } else {
        let ctx = Arc::new(CudaContextWrapper::new(device_id).map_err(|e| e.to_string())?);
        contexts[device_id] = Some(ctx.clone());
        Ok(ctx)
    }
}

pub fn get_cuda_device_count() -> Result<usize, DriverError> {
    use cudarc::driver::result;
    result::init()?;
    let count = result::device::get_count()?;
    Ok(count as usize)
}

// 检测 CUDA 可用性
pub fn cuda_available() -> bool {
    match get_cuda_device_count() {
        Ok(count) if count > 0 => true,
        _ => {
            eprintln!("Skipping CUDA test: no device found");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cuda_device_count() {
        // 即使没有 CUDA，函数也应返回 Ok(0) 或 Err，不应 panic
        let result = get_cuda_device_count();
        if cuda_available() {
            assert!(result.is_ok());
            assert!(result.unwrap() > 0);
        } else {
            // 没有 CUDA 时可能返回 Ok(0) 或 Err，都接受
            println!("CUDA not available, count result: {:?}", result);
        }
    }

    #[test]
    fn test_set_and_get_current_device() {
        // 测试 CPU 设备（索引0）的当前设备设置
        set_current_device(0);
        assert_eq!(get_current_device(), Some(0));
        set_current_device(1);
        assert_eq!(get_current_device(), Some(1));
        // 重置
        set_current_device(0);
    }

    #[test]
    fn test_get_or_create_context() {
        if !cuda_available() {
            return;
        }
        let ctx = get_or_create_context(0).expect("Failed to get or create context");
        assert_eq!(ctx.device_id, 0);
        // 再次获取应返回相同的 Arc
        let ctx2 = get_or_create_context(0).expect("Failed to get context again");
        assert!(Arc::ptr_eq(&ctx, &ctx2));
    }

    #[test]
    fn test_multiple_contexts() {
        if !cuda_available() {
            return;
        }
        let device_count = get_cuda_device_count().unwrap();
        if device_count < 2 {
            eprintln!(
                "Skipping multi-context test: only {} device(s)",
                device_count
            );
            return;
        }
        let ctx0 = get_or_create_context(0).unwrap();
        let ctx1 = get_or_create_context(1).unwrap();
        assert_ne!(ctx0.device_id, ctx1.device_id);
        // 确保不同设备的上下文不同
        assert!(!Arc::ptr_eq(&ctx0, &ctx1));
    }

    #[test]
    fn test_cuda_context_wrapper_stream_ptr() {
        if !cuda_available() {
            return;
        }
        let wrapper = CudaContextWrapper::new(0).expect("Failed to create wrapper");
        let stream_ptr = wrapper.stream_ptr();
        let expected = wrapper.stream.cu_stream();
        assert_eq!(stream_ptr, expected as *mut std::ffi::c_void);
    }
}
