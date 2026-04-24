use cudarc::driver::{CudaContext, CudaStream, DriverError};
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

thread_local! {
    static CURRENT_CUDA_DEVICE: RefCell<Option<usize>> = const { RefCell::new(None) };
}

pub fn set_device(device_id: usize) -> Result<(), String> {
    let count = get_device_count()?;
    if device_id >= count {
        return Err(format!(
            "Invalid device id {}, only {} devices available",
            device_id, count
        ));
    }
    CURRENT_CUDA_DEVICE.with(|d| *d.borrow_mut() = Some(device_id));
    Ok(())
}

pub fn get_device() -> usize {
    CURRENT_CUDA_DEVICE.with(|d| {
        if let Some(id) = *d.borrow() {
            id
        } else {
            let default_id = 0;
            *d.borrow_mut() = Some(default_id);
            default_id
        }
    })
}

pub fn get_device_count() -> Result<usize, String> {
    use cudarc::driver::result;
    result::init().map_err(|e| e.to_string())?;
    let count = result::device::get_count().map_err(|e| e.to_string())?;
    Ok(count as usize) // 转换为 usize
}

pub fn is_available() -> bool {
    get_device_count().unwrap_or(0) > 0
}

pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub device_id: usize,
}

impl DeviceContext {
    pub fn new(device_id: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?; // 直接传播 DriverError
        Ok(DeviceContext { ctx, device_id })
    }

    pub fn create_stream(&self) -> Result<Arc<CudaStream>, DriverError> {
        self.ctx.new_stream()
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }
}

static DEVICE_CONTEXTS: Lazy<Mutex<HashMap<usize, Arc<DeviceContext>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_device_context(device_id: usize) -> Result<Arc<DeviceContext>, String> {
    let mut map = DEVICE_CONTEXTS.lock().unwrap();
    if let Some(ctx) = map.get(&device_id) {
        Ok(ctx.clone())
    } else {
        let ctx = Arc::new(DeviceContext::new(device_id).map_err(|e| e.to_string())?);
        map.insert(device_id, ctx.clone());
        Ok(ctx)
    }
}
