use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc; // 添加

#[derive(Clone, Debug)]
pub(crate) enum DataPtr {
    Cpu(Box<[u8]>),
    Gpu(CudaSlice<u8>),
}

impl DataPtr {
    pub(crate) fn as_ptr(&self, stream: Option<&Arc<CudaStream>>) -> *const u8 {
        match self {
            DataPtr::Cpu(b) => b.as_ptr(),
            DataPtr::Gpu(s) => {
                let stream_ref = stream.expect("Stream required for GPU pointer");
                let (ptr, _sync) = s.device_ptr(stream_ref);
                ptr as *const u8
            }
        }
    }
    pub(crate) fn as_mut_ptr(&mut self, stream: Option<&Arc<CudaStream>>) -> *mut u8 {
        match self {
            DataPtr::Cpu(b) => b.as_mut_ptr(),
            DataPtr::Gpu(s) => {
                let stream_ref = stream.expect("Stream required for GPU pointer");
                let (ptr, _sync) = s.device_ptr_mut(stream_ref);
                ptr as *mut u8
            }
        }
    }
    pub(crate) fn is_gpu(&self) -> bool {
        matches!(self, DataPtr::Gpu(_))
    }
}
