use anyhow::{Context, Result};
use cudarc::driver::CudaEvent;
use std::time::Duration;

pub struct Event {
    pub(crate) event: CudaEvent,
}

impl Event {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = super::device::get_device_context(device_id)?;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let event = ctx
            .ctx
            .new_event(flags)
            .context("Failed to create CUDA event")?;
        Ok(Event { event })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event
            .synchronize()
            .context("Failed to synchronize CUDA event")
    }

    pub fn done(&self) -> bool {
        self.event.is_complete()
    }

    pub fn elapsed_since(&self, earlier: &Self) -> Result<Duration> {
        let ms = self
            .event
            .elapsed_ms(&earlier.event)
            .context("Failed to get elapsed time")?;
        let ms_abs = ms.abs();
        let secs = ms_abs as f64 / 1000.0;
        Ok(Duration::from_secs_f64(secs))
    }
}
