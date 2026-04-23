mod device;
mod event;
mod stream;

pub use device::{get_device, set_device};
pub use event::Event;
pub use stream::{default_stream, get_stream, set_stream, Stream}; // 移除 new_event，因为 Event 有 new()
