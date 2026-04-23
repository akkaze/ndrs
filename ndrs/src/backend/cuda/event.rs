use cudarc::driver::CudaEvent;
use std::time::Duration;

pub struct Event {
    pub(crate) event: CudaEvent,
}

impl Event {
    pub fn new(device_id: usize) -> Result<Self, String> {
        let ctx = super::device::get_device_context(device_id)?;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let event = ctx.ctx.new_event(flags).map_err(|e| e.to_string())?;
        Ok(Event { event })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.event.synchronize().map_err(|e| e.to_string())
    }

    pub fn done(&self) -> bool {
        self.event.is_complete()
    }

    pub fn elapsed_since(&self, earlier: &Self) -> Result<Duration, String> {
        let ms = self
            .event
            .elapsed_ms(&earlier.event)
            .map_err(|e| e.to_string())?;
        Ok(Duration::from_secs_f64(ms as f64 / 1000.0))
    }
}
