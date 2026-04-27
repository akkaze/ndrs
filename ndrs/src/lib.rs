// ndrs/src/lib.rs
pub mod backend;
pub mod builtin_kernels;
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

pub use builtin_kernels::cpu as cpu_kernels;
pub use builtin_kernels::cuda as cuda_kernels;

pub use device::Device;
pub use dtype::{
    BinaryOpFn, BinaryOpKind, DTYPE_FLOAT32, DTYPE_INT32, DType, TypeInfo, allocate_dtype,
    get_add_op, get_binary_op, register_binary_op, register_dtype,
};
pub use tensor::{ArcTensor, RcTensor, Tensor};
pub use view::{ArcTensorView, RcTensorView, SliceInfo, TensorViewOps, broadcast_shapes};

#[cfg(test)]
mod test_init {
    use ctor::ctor;

    #[ctor]
    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
}
