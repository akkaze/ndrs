//! 张量视图基础模块

mod broadcast;
mod slice;
mod trait_def; // 改名为 trait_def 避免与关键字冲突
mod views;

pub use broadcast::broadcast_shapes;
pub use slice::{SliceArg, SliceInfo};
pub use trait_def::TensorViewOps;
pub use views::{arc_view_to_vec_f32, rc_view_to_vec_f32};
pub use views::{ArcTensorView, AsView, RcTensorView};
