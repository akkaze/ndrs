mod device;
mod event;
mod stream;

pub use device::{get_device, set_device};
pub use event::Event;
pub use stream::{Stream, default_stream, get_stream, set_stream}; // 移除 new_event，因为 Event 有 new()
