use super::device::get_device as is_cpu_device;
use super::event::Event;
use anyhow::{Context, Result, anyhow, bail};
use std::sync::Arc;
use threadpool::ThreadPool;

#[derive(Clone)]
struct CpuStreamInner {
    pool: ThreadPool,
}

impl CpuStreamInner {
    fn new(num_threads: usize) -> Self {
        CpuStreamInner {
            pool: ThreadPool::new(num_threads),
        }
    }

    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.execute(f);
    }

    fn synchronize(&self) {
        self.pool.join();
    }
}

#[derive(Clone)]
pub struct Stream {
    inner: Arc<CpuStreamInner>,
}

impl Stream {
    pub fn new() -> Self {
        let inner = Arc::new(CpuStreamInner::new(num_cpus::get()));
        Stream { inner }
    }

    pub fn synchronize(&self) {
        self.inner.synchronize();
    }

    pub fn wait_event(&self, event: &Event) {
        event.synchronize(); // 等待事件完成即可
    }

    pub fn record(&self) -> Event {
        let event = Event::new();
        let inner = self.inner.clone();
        let event_clone = event.clone();
        inner.execute(move || {
            // 空操作，仅用于记录完成时间
            event_clone.synchronize(); // 实际上事件在创建时就已完成？需要更精确
        });
        event
    }
}

static DEFAULT_STREAM: std::sync::OnceLock<Stream> = std::sync::OnceLock::new();

pub fn default_stream() -> &'static Stream {
    DEFAULT_STREAM.get_or_init(|| Stream::new())
}

thread_local! {
    static CURRENT_STREAM: std::cell::RefCell<Option<Stream>> = const { std::cell::RefCell::new(None) };
}

pub fn set_stream(stream: Stream) -> anyhow::Result<()> {
    if !is_cpu_device() {
        anyhow::bail!("Cannot set CPU stream when current device is not CPU");
    }
    CURRENT_STREAM.with(|s| *s.borrow_mut() = Some(stream));
    Ok(())
}

pub fn get_stream() -> Stream {
    CURRENT_STREAM.with(|s| {
        s.borrow()
            .clone()
            .unwrap_or_else(|| default_stream().clone())
    })
}
