use super::device::{get_device, get_device_context};
use super::event::Event;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::CudaStream;
use std::sync::Arc;

#[derive(Clone)]
pub struct Stream {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) device_id: usize,
}

impl Stream {
    pub fn new(device_id: Option<usize>) -> Result<Self, String> {
        let dev_id = device_id.unwrap_or_else(|| get_device());
        let ctx = get_device_context(dev_id)?;
        let stream = ctx.create_stream().map_err(|e| e.to_string())?;
        Ok(Stream {
            stream,
            device_id: dev_id,
        })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|e| e.to_string())
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), String> {
        self.stream.wait(&event.event).map_err(|e| e.to_string())
    }

    pub fn record(&self) -> Result<Event, String> {
        let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
        // 直接记录事件，即使流句柄为 0 也是允许的（默认流）
        let event = self.stream.record_event(flags).map_err(|e| e.to_string())?;
        Ok(Event { event })
    }

    pub fn join(&self, other: &Stream) -> Result<(), String> {
        self.stream.join(&other.stream).map_err(|e| e.to_string())
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.stream.cu_stream() as *mut _
    }

    pub fn inner(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

pub fn default_stream() -> Result<Stream, String> {
    let dev_id = get_device();
    let ctx = get_device_context(dev_id)?;
    let stream = ctx.default_stream();
    // 默认流句柄可以是 0，是合法值，无需断言
    Ok(Stream {
        stream,
        device_id: dev_id,
    })
}
thread_local! {
    static CURRENT_STREAM: std::cell::RefCell<Option<Stream>> = const { std::cell::RefCell::new(None) };
}

pub fn set_stream(stream: Stream) -> Result<(), String> {
    let current_dev = get_device();
    if stream.device_id != current_dev {
        return Err(format!(
            "Stream device {} does not match current CUDA device {}",
            stream.device_id, current_dev
        ));
    }
    CURRENT_STREAM.with(|s| *s.borrow_mut() = Some(stream));
    Ok(())
}

pub fn get_stream() -> Result<Stream, String> {
    CURRENT_STREAM.with(|s| {
        if let Some(stream) = s.borrow().clone() {
            Ok(stream)
        } else {
            default_stream()
        }
    })
}
