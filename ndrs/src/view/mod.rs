//! 张量视图基础模块

mod slice;
mod broadcast;
mod trait_def;  // 改名为 trait_def 避免与关键字冲突
mod views;

pub use slice::{SliceArg, SliceInfo};
pub use broadcast::broadcast_shapes;
pub use trait_def::TensorViewOps;
pub use views::{RcTensorView, ArcTensorView, AsView};
pub use views::{rc_view_to_vec_f32, arc_view_to_vec_f32};