use crate::device::{get_current_device, CudaContextWrapper};
use cudarc::driver::sys::CUevent_flags; // 只导入一次
use cudarc::driver::CudaEvent;
use std::sync::Arc;

pub struct Stream {
    pub(crate) ctx: Arc<CudaContextWrapper>,
    pub(crate) stream: Arc<cudarc::driver::CudaStream>,
}

impl Stream {
    pub fn new() -> Result<Self, String> {
        eprintln!("[DEBUG Stream::new] called");
        let dev_id = get_current_device().ok_or("No current device")?;
        eprintln!("[DEBUG Stream::new] device_id = {}", dev_id);
        let ctx = Arc::new(CudaContextWrapper::new(dev_id).map_err(|e| e.to_string())?);
        let stream = ctx.ctx.default_stream();
        eprintln!("[DEBUG Stream::new] stream created");
        Ok(Stream { ctx, stream })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|e| e.to_string())
    }

    /// 让当前流等待一个事件完成
    pub fn wait_event(&self, event: &Event) -> Result<(), String> {
        self.stream.wait(&event.event).map_err(|e| e.to_string())
    }

    /// 让当前流等待另一个流的所有待处理工作完成
    pub fn join(&self, other: &Stream) -> Result<(), String> {
        let event = other.record()?;
        self.wait_event(&event)
    }

    /// 在当前流中记录一个事件
    pub fn record(&self) -> Result<Event, String> {
        let flags: Option<CUevent_flags> = None; // 默认 CU_EVENT_DEFAULT
        let event = self.stream.record_event(flags).map_err(|e| e.to_string())?;
        Ok(Event { event })
    }
}

pub struct Event {
    event: CudaEvent,
}

impl Event {
    /// 创建新事件（未记录到任何流）
    pub fn new_with_context(ctx: &CudaContextWrapper) -> Result<Self, String> {
        eprintln!("[DEBUG Event::new_with_context] called");
        let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
        let event = ctx.ctx.new_event(flags).map_err(|e| e.to_string())?;
        eprintln!("[DEBUG Event::new_with_context] event created");
        Ok(Event { event })
    }

    /// 同步等待事件完成
    pub fn synchronize(&self) -> Result<(), String> {
        self.event.synchronize().map_err(|e| e.to_string())
    }

    /// 检查事件是否已完成（非阻塞）
    pub fn done(&self) -> bool {
        self.event.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{self, cuda_available, get_cuda_device_count, set_current_device};

    #[test]
    fn test_stream_sync() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream = Stream::new().expect("Failed to create stream");
        stream.synchronize().expect("Failed to synchronize stream");
    }

    #[test]
    fn test_event_record_and_sync() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream = Stream::new().expect("Failed to create stream");
        let event = stream.record().expect("Failed to record event");
        event.synchronize().expect("Failed to synchronize event");
    }

    #[test]
    fn test_event_done() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream = Stream::new().expect("Failed to create stream");
        let event = stream.record().expect("Failed to record event");
        // 事件可能尚未完成，但同步后应该完成
        event.synchronize().expect("Failed to synchronize event");
        assert!(event.done(), "Event should be done after synchronization");
    }

    #[test]
    fn test_stream_join() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream1 = Stream::new().expect("Failed to create stream1");
        let stream2 = Stream::new().expect("Failed to create stream2");
        // 在 stream2 中记录一个事件（实际可以做一些工作，但记录事件本身足够）
        let _event = stream2.record().expect("Failed to record event in stream2");
        // 让 stream1 等待 stream2 完成
        stream1.join(&stream2).expect("Failed to join streams");
        stream1
            .synchronize()
            .expect("Failed to synchronize stream1");
    }

    #[test]
    fn test_event_creation_with_context() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let ctx = device::get_or_create_context(0).expect("Failed to get CUDA context");
        let event = Event::new_with_context(&ctx).expect("Failed to create event with context");
        // 事件未记录时，done 应为 false（取决于实现，默认未记录时可能为 false）
        // 注意：新创建但未记录的事件，is_complete() 的行为未定义，所以不测试 done
        event
            .synchronize()
            .expect("Failed to synchronize unrecorded event");
    }

    #[test]
    fn test_wait_event() {
        if !cuda_available() {
            return;
        }
        set_current_device(0);
        let stream1 = Stream::new().expect("Failed to create stream1");
        let stream2 = Stream::new().expect("Failed to create stream2");
        let event = stream2.record().expect("Failed to record event in stream2");
        stream1.wait_event(&event).expect("Failed to wait on event");
        stream1
            .synchronize()
            .expect("Failed to synchronize stream1");
    }
}
