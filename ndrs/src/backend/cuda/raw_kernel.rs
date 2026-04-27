use anyhow::{Context, Result, anyhow, bail};
use cudarc::driver::{CudaContext, CudaStream, LaunchArgs, LaunchConfig};
use cudarc::nvrtc::{Ptx, compile_ptx};
use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// 封装的 CUDA 内核，支持动态编译和启动。
/// 用法：
/// ```ignore
/// let kernel = RawKernel::from_source(src, "my_kernel", &ctx)?;
/// let mut builder = kernel.launch_builder(&stream);
/// builder.arg(&arg1);
/// builder.arg(&arg2);
/// unsafe { builder.launch(cfg)?; }
/// ```
pub struct RawKernel {
    module: Arc<cudarc::driver::CudaModule>,
    func: cudarc::driver::CudaFunction,
}

impl RawKernel {
    /// 从 PTX 对象加载内核
    pub fn from_ptx(ptx: &Ptx, name: &str, ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = ctx.load_module(ptx.clone())?;
        let func = module.load_function(name)?;
        Ok(RawKernel { module, func })
    }

    /// 从 CUDA C++ 源代码编译内核
    pub fn from_source(src: &str, name: &str, ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let ptx = compile_ptx(src)?;
        Self::from_ptx(&ptx, name, ctx)
    }

    /// 返回一个 `LaunchArgs` builder，可以直接添加参数并启动
    pub fn launch_builder<'a>(&'a self, stream: &'a CudaStream) -> LaunchArgs<'a> {
        stream.launch_builder(&self.func)
    }
}

/// 启动 RawKernel 的宏，自动添加参数。
/// 用法: `launch!(kernel, stream, cfg, arg1, arg2, ...);`
#[macro_export]
macro_rules! launch {
    ($kernel:expr, $stream:expr, $cfg:expr, $($arg:expr),*) => {{
        let mut builder = $stream.launch_builder(&$kernel.func);
        $(builder.arg($arg);)*
        unsafe { builder.launch($cfg)?; }
        Ok(())
    }};
}
