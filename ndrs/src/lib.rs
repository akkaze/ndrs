pub mod backend;
pub mod device;
pub mod dtype;
pub mod kernel;
pub mod macros;
pub mod stream;
pub mod tensor;
pub mod view;
pub mod view_ops;

pub use backend::cpu;
pub use backend::cuda;

// 统一设备/流 API
pub use device::Device;

// 其他导出保持不变
pub use dtype::{DType, DTYPE_FLOAT32, DTYPE_INT32};
pub use tensor::Tensor;
pub use view::{broadcast_shapes, ArcTensorView, RcTensorView, SliceInfo, TensorViewOps};

#[cfg(test)]
mod test_init {
    use ctor::ctor;

    #[ctor]
    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
}
