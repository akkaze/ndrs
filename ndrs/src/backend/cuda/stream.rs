use super::device::{get_device, get_device_context};
use super::event::Event;
use anyhow::{Context, Result, bail};
use cudarc::driver::CudaStream;
use cudarc::driver::sys::CUevent_flags;
use std::sync::Arc;

#[derive(Clone)]
pub struct Stream {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) device_id: usize,
}

impl Stream {
    pub fn new(device_id: Option<usize>) -> Result<Self> {
        let dev_id = device_id.unwrap_or_else(|| get_device());
        let ctx = get_device_context(dev_id).context("Failed to get device context")?;
        let stream = ctx
            .create_stream()
            .context("Failed to create CUDA stream")?;
        Ok(Stream {
            stream,
            device_id: dev_id,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("Failed to synchronize CUDA stream")
    }

    pub fn wait_event(&self, event: &Event) -> Result<()> {
        self.stream
            .wait(&event.event)
            .context("Failed to wait on CUDA event")
    }

    pub fn record(&self) -> Result<Event> {
        let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
        let event = self
            .stream
            .record_event(flags)
            .context("Failed to record CUDA event")?;
        Ok(Event { event })
    }

    pub fn join(&self, other: &Stream) -> Result<()> {
        self.stream
            .join(&other.stream)
            .context("Failed to join CUDA streams")
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.stream.cu_stream() as *mut _
    }

    pub fn inner(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

pub fn default_stream() -> Result<Stream> {
    let dev_id = get_device();
    let ctx = get_device_context(dev_id).context("Failed to get device context")?;
    let stream = ctx.default_stream();
    Ok(Stream {
        stream,
        device_id: dev_id,
    })
}

thread_local! {
    static CURRENT_STREAM: std::cell::RefCell<Option<Stream>> = const { std::cell::RefCell::new(None) };
}

pub fn set_stream(stream: Stream) -> Result<()> {
    let current_dev = get_device();
    if stream.device_id != current_dev {
        bail!(
            "Stream device {} does not match current CUDA device {}",
            stream.device_id,
            current_dev
        );
    }
    CURRENT_STREAM.with(|s| *s.borrow_mut() = Some(stream));
    Ok(())
}

pub fn get_stream() -> Result<Stream> {
    CURRENT_STREAM.with(|s| {
        if let Some(stream) = s.borrow().clone() {
            Ok(stream)
        } else {
            default_stream()
        }
    })
}
