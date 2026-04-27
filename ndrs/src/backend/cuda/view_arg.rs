// ndrs/src/backend/cuda/view_arg.rs
use crate::view::{ArcTensorView, RcTensorView};
use cudarc::driver::{CudaSlice, LaunchArgs, PushKernelArg};

// macro_rules! impl_push_kernel_arg_for_view {
//     ($view:ty, $get_slice:ident, $get_slice_mut:ident) => {
//         unsafe impl<'a, 'b: 'a> PushKernelArg<&'b $view> for LaunchArgs<'a> {
//             #[inline]
//             fn arg(&mut self, arg: &'b $view) -> &mut Self {
//                 let slice = unsafe { $get_slice(arg) };
//                 self.arg(slice)
//             }
//         }

//         unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut $view> for LaunchArgs<'a> {
//             #[inline]
//             fn arg(&mut self, arg: &'b mut $view) -> &mut Self {
//                 let slice = unsafe { $get_slice_mut(arg) };
//                 self.arg(slice)
//             }
//         }
//     };
// }

// impl_push_kernel_arg_for_view!(RcTensorView, get_rc_gpu_slice, get_rc_gpu_slice_mut);
// impl_push_kernel_arg_for_view!(ArcTensorView, get_arc_gpu_slice, get_arc_gpu_slice_mut);
