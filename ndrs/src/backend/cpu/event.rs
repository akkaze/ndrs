use anyhow::{Context, Result, anyhow, bail};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Event {
    inner: Arc<CpuEventInner>,
}

#[derive(Clone)]
struct CpuEventInner {
    completed: Arc<AtomicUsize>,
    condvar: Arc<Condvar>,
    mutex: Arc<Mutex<()>>,
    timestamp: Arc<Mutex<Option<Instant>>>,
}

impl CpuEventInner {
    fn new() -> Self {
        CpuEventInner {
            completed: Arc::new(AtomicUsize::new(0)),
            condvar: Arc::new(Condvar::new()),
            mutex: Arc::new(Mutex::new(())),
            timestamp: Arc::new(Mutex::new(None)),
        }
    }

    fn complete(&self) {
        let now = Instant::now();
        *self.timestamp.lock().unwrap() = Some(now);
        self.completed.store(1, Ordering::SeqCst);
        self.condvar.notify_all();
    }

    fn wait(&self) {
        let mut guard = self.mutex.lock().unwrap();
        while self.completed.load(Ordering::SeqCst) == 0 {
            guard = self.condvar.wait(guard).unwrap();
        }
    }

    fn done(&self) -> bool {
        self.completed.load(Ordering::SeqCst) == 1
    }

    fn timestamp(&self) -> Option<Instant> {
        *self.timestamp.lock().unwrap()
    }
}

impl Event {
    pub fn new() -> Self {
        Event {
            inner: Arc::new(CpuEventInner::new()),
        }
    }

    pub fn synchronize(&self) {
        self.inner.wait();
    }

    pub fn done(&self) -> bool {
        self.inner.done()
    }

    pub fn elapsed_since(&self, earlier: &Self) -> anyhow::Result<std::time::Duration> {
        use anyhow::{anyhow, bail};
        let t1 = self
            .inner
            .timestamp()
            .ok_or_else(|| anyhow!("Event not completed"))?;
        let t2 = earlier
            .inner
            .timestamp()
            .ok_or_else(|| anyhow!("Earlier event not completed"))?;
        if t1 < t2 {
            bail!("Current event occurred before earlier event");
        }
        Ok(t1 - t2)
    }
}
