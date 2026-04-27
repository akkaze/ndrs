use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaStream, DriverError};
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

thread_local! {
    static CURRENT_CUDA_DEVICE: RefCell<Option<usize>> = const { RefCell::new(None) };
}

pub fn set_device(device_id: usize) -> Result<()> {
    let count = get_device_count()?;
    if device_id >= count {
        anyhow::bail!(
            "Invalid device id {}, only {} devices available",
            device_id,
            count
        );
    }
    CURRENT_CUDA_DEVICE.with(|d| *d.borrow_mut() = Some(device_id));
    Ok(())
}

pub fn get_device() -> usize {
    CURRENT_CUDA_DEVICE.with(|d| {
        let mut borrowed = d.borrow_mut();
        if let Some(id) = *borrowed {
            id
        } else {
            *borrowed = Some(0);
            0
        }
    })
}

pub fn get_device_count() -> Result<usize> {
    use cudarc::driver::result;
    result::init().context("Failed to init CUDA driver")?;
    let count = result::device::get_count().context("Failed to get device count")?;
    Ok(count as usize)
}

pub fn is_available() -> bool {
    get_device_count().unwrap_or(0) > 0
}

pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub device_id: usize,
}

impl DeviceContext {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id)?;
        ctx.bind_to_thread()?;
        Ok(DeviceContext { ctx, device_id })
    }

    pub fn create_stream(&self) -> Result<Arc<CudaStream>> {
        self.ctx
            .new_stream()
            .context("Failed to create CUDA stream")
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }
}

static DEVICE_CONTEXTS: Lazy<Mutex<HashMap<usize, Arc<DeviceContext>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_device_context(device_id: usize) -> Result<Arc<DeviceContext>> {
    let mut map = DEVICE_CONTEXTS
        .lock()
        .map_err(|_| anyhow!("Lock poisoned"))?;
    if let Some(ctx) = map.get(&device_id) {
        Ok(ctx.clone())
    } else {
        let ctx =
            Arc::new(DeviceContext::new(device_id).context("Failed to create device context")?);
        map.insert(device_id, ctx.clone());
        Ok(ctx)
    }
}
