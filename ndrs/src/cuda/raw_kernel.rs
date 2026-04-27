/// 动态 CUDA 内核启动器（支持直接传递 Tensor）

use crate::Tensor;
use crate::tensor::DataPtr;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// 表示一个已编译的 CUDA 内核
pub struct RawKernel {
    module: Arc<cudarc::driver::CudaModule>,
    func: cudarc::driver::CudaFunction,
    name: String,
}

impl RawKernel {
    // ... 其他方法保持不变 (from_ptx, from_source) ...
    /// 从 PTX 代码加载内核
    pub fn from_ptx(ptx: &str, name: &str, ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let module = ctx.load_module_ptx(ptx).map_err(|e| e.to_string())?;
        let func = module.load_function(name).map_err(|e| e.to_string())?;
        Ok(RawKernel {
            module,
            func,
            name: name.to_string(),
        })
    }

    /// 从 CUDA C++ 源代码编译内核
    pub fn from_source(src: &str, name: &str, ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(src).map_err(|e| e.to_string())?;
        Self::from_ptx(&ptx, name, ctx)
    }
    /// 启动内核，参数可以是实现了 `KernelArg` 的任何类型。
    pub fn launch(
        &self,
        stream: &CudaStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[&dyn KernelArg],
        shared_mem: u32,
    ) -> Result<(), String> {
        let mut builder = stream.launch_builder(&self.func);
        for arg in args {
            builder = arg.add_to_builder(builder);
        }
        let config = LaunchConfig {
            grid_dim: (grid.0, grid.1, grid.2),
            block_dim: (block.0, block.1, block.2),
            shared_mem_bytes: shared_mem,
        };
        unsafe {
            builder.launch(config).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// 可添加到 `LaunchBuilder` 的参数 trait
pub trait KernelArg {
    fn add_to_builder<'a>(
        &self,
        builder: cudarc::driver::LaunchBuilder<'a>,
    ) -> cudarc::driver::LaunchBuilder<'a>;
}

// 为常见标量类型实现
macro_rules! impl_kernel_arg_scalar {
    ($t:ty) => {
        impl KernelArg for $t {
            fn add_to_builder<'a>(
                &self,
                builder: cudarc::driver::LaunchBuilder<'a>,
            ) -> cudarc::driver::LaunchBuilder<'a> {
                builder.arg(self)
            }
        }
    };
}
impl_kernel_arg_scalar!(i32);
impl_kernel_arg_scalar!(f32);
impl_kernel_arg_scalar!(usize);
// 可以为 u32, f64 等扩展

impl<T: cudarc::driver::DeviceRepr> KernelArg for &mut CudaSlice<T> {
    fn add_to_builder<'a>(
        &self,
        builder: cudarc::driver::LaunchBuilder<'a>,
    ) -> cudarc::driver::LaunchBuilder<'a> {
        builder.arg(*self)
    }
}

impl<T: cudarc::driver::DeviceRepr> KernelArg for &CudaSlice<T> {
    fn add_to_builder<'a>(
        &self,
        builder: cudarc::driver::LaunchBuilder<'a>,
    ) -> cudarc::driver::LaunchBuilder<'a> {
        builder.arg(*self)
    }
}

// 为 &mut Tensor（GPU 上）实现 KernelArg
impl KernelArg for &mut Tensor {
    fn add_to_builder<'a>(
        &self,
        builder: cudarc::driver::LaunchBuilder<'a>,
    ) -> cudarc::driver::LaunchBuilder<'a> {
        match &mut self.data {
            DataPtr::Gpu(slice) => builder.arg(slice),
            DataPtr::Cpu(_) => panic!("Cannot pass CPU tensor to CUDA kernel"),
        }
    }
}

// 为 &Tensor（只读）实现
impl KernelArg for &Tensor {
    fn add_to_builder<'a>(
        &self,
        builder: cudarc::driver::LaunchBuilder<'a>,
    ) -> cudarc::driver::LaunchBuilder<'a> {
        match &self.data {
            DataPtr::Gpu(slice) => builder.arg(slice),
            DataPtr::Cpu(_) => panic!("Cannot pass CPU tensor to CUDA kernel"),
        }
    }
}

/// 用常数值填充 GPU Tensor (f32)
pub fn fill_tensor_f32_const(
    tensor: &mut Tensor,
    value: f32,
    stream: &CudaStream,
    ctx: &Arc<CudaContext>,
) -> Result<(), String> {
    let n = tensor.size();
    if n == 0 {
        return Ok(());
    }
    let kernel_src = r#"
        extern "C" __global__ void fill_f32(float* out, const int n, const float val) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = val;
        }
    "#;
    let kernel = RawKernel::from_source(kernel_src, "fill_f32", ctx)?;
    let block = 256;
    let grid = (n + block - 1) / block;
    kernel.launch(
        stream,
        (grid as u32, 1, 1),
        (block, 1, 1),
        &[tensor, &n, &value], // 直接传递 &mut Tensor
        0,
    )?;
    Ok(())
}

/// 填充为零（memset 更快）
pub fn fill_tensor_f32_zero(tensor: &mut Tensor, stream: &CudaStream) -> Result<(), String> {
    let slice = match &mut tensor.data {
        DataPtr::Gpu(s) => s,
        _ => return Err("Tensor not on GPU".into()),
    };
    let num_bytes = slice.len() * std::mem::size_of::<f32>();
    let (ptr, _) = slice.device_ptr_mut(stream);
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr as *mut u8, 0, num_bytes, stream.cu_stream())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
