// ndrs/src/backend/cuda/tensor_arg.rs
use crate::tensor::{DataPtr, Tensor};
use cudarc::driver::{LaunchArgs, PushKernelArg};

// 为 &Tensor 实现 PushKernelArg
unsafe impl<'a, 'b: 'a> PushKernelArg<&'b Tensor> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b Tensor) -> &mut Self {
        match &arg.data {
            DataPtr::Gpu(slice) => {
                // slice: &CudaSlice<u8>
                // 调用已有的 PushKernelArg<&CudaSlice<u8>> 实现
                self.arg(slice)
            }
            DataPtr::Cpu(_) => panic!("Cannot pass CPU tensor to CUDA kernel"),
        }
    }
}

// 为 &mut Tensor 实现 PushKernelArg
unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut Tensor> for LaunchArgs<'a> {
    #[inline]
    fn arg(&mut self, arg: &'b mut Tensor) -> &mut Self {
        match &mut arg.data {
            DataPtr::Gpu(slice) => {
                // slice: &mut CudaSlice<u8>
                self.arg(slice)
            }
            DataPtr::Cpu(_) => panic!("Cannot pass mutable CPU tensor to CUDA kernel"),
        }
    }
}
