pub mod device;
pub mod dtype;
pub mod kernel;
pub mod stream;
pub mod tensor;
pub mod view;
pub mod view_ops;

// 重新导出主要类型
pub use device::{CudaContextWrapper as CudaContext, Device};
pub use dtype::{
    get_add_op, get_dtype_info, register_add_op, register_dtype, DType, DTypeMapping,
    DTYPE_FLOAT32, DTYPE_INT32,
};
pub use stream::{Event, Stream};
pub use tensor::Tensor;
pub use view::{broadcast_shapes, ArcTensorView, AsView, RcTensorView, SliceArg, SliceInfo, TensorViewOps};
pub use ndrs_macros::s;
pub use view::{rc_view_to_vec_f32, arc_view_to_vec_f32};
