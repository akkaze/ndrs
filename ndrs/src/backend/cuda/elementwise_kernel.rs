use crate::Device;
use crate::backend::cuda::RawKernel;
use crate::cuda::get_stream;
use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};
use crate::tensor::Tensor;
use crate::view::TensorViewOps;
use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::Arc;
pub struct ElementwiseKernel {
    base_name: String,
    kernels: HashMap<String, Arc<RawKernel>>,
    ctx: Arc<CudaContext>,
    params: Vec<(String, String, bool)>,
    expr: String,
}

impl ElementwiseKernel {
    pub fn from_expression(
        params_str: &str,
        expr: &str,
        name: &str,
        ctx: Arc<CudaContext>,
    ) -> Result<Self> {
        let eq_pos = expr.find('=').context("Missing '='")?;
        let out_var = expr[0..eq_pos].trim().to_string();
        let body = expr[eq_pos + 1..].trim();
        let expr = format!("{} = {}", out_var, body);

        let mut params = Vec::new();
        for part in params_str.split(',') {
            let part = part.trim();
            let tokens: Vec<&str> = part.split_whitespace().collect();
            if tokens.len() != 2 {
                bail!("Invalid param: {}", part);
            }
            let placeholder = tokens[0].to_string();
            let name = tokens[1].to_string();
            let is_output = name == out_var;
            params.push((placeholder, name, is_output));
        }

        Ok(ElementwiseKernel {
            base_name: name.to_string(),
            kernels: HashMap::new(),
            ctx,
            params,
            expr,
        })
    }
    fn get_or_compile_kernel(
        &mut self,
        dtype_ids: &[u32],
        ndim: usize,
        _shape: &[usize],
    ) -> Result<Arc<RawKernel>> {
        let dtype_key = dtype_ids
            .iter()
            .map(|&id| id.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let key = format!("{}_{}_{}", self.base_name, dtype_key, ndim);
        if let Some(k) = self.kernels.get(&key) {
            return Ok(k.clone());
        }

        // 建立占位符到 C 类型的映射
        let mut type_map = HashMap::new();
        for (i, (placeholder, _, _)) in self.params.iter().enumerate() {
            if !type_map.contains_key(placeholder) {
                let dtype_id = dtype_ids[i];
                let c_type = match dtype_id {
                    DTYPE_FLOAT32 => "float",
                    DTYPE_INT32 => "int",
                    _ => bail!("Unsupported dtype: {}", dtype_id),
                };
                type_map.insert(placeholder.clone(), c_type);
            }
        }

        // 构建内核参数列表：所有数据指针均为 unsigned char*
        let mut kernel_args = String::new();
        for (placeholder, name, is_output) in &self.params {
            let qualifier = if *is_output { "" } else { "const " };
            kernel_args.push_str(&format!(
                "{qualifier}unsigned char* data_{name}, const size_t* strides_{name}, ",
            ));
        }
        kernel_args.push_str("const size_t* shape, const int ndim, const size_t total_elements");

        // 偏移计算代码
        let mut offset_calc = String::new();
        for (_, name, _) in &self.params {
            offset_calc.push_str(&format!("size_t off_{} = 0; ", name));
        }
        offset_calc.push_str(
        "size_t temp = idx; for (int d = ndim - 1; d >= 0; --d) { size_t i = temp % shape[d]; temp /= shape[d]; "
    );
        for (_, name, _) in &self.params {
            offset_calc.push_str(&format!("off_{} += i * strides_{}[d]; ", name, name));
        }
        offset_calc.push_str("}");

        // 表达式替换：将变量名替换为带类型转换的访存
        let mut body = self.expr.clone();
        for (_, name, is_output) in &self.params {
            let placeholder = self
                .params
                .iter()
                .find(|(_, n, _)| n == name)
                .unwrap()
                .0
                .clone();
            let c_type = type_map[&placeholder];
            let replacement = if *is_output {
                format!("*(({}*)(data_{} + off_{}))", c_type, name, name)
            } else {
                format!("*((const {}*)(data_{} + off_{}))", c_type, name, name)
            };
            body = body.replace(name, &replacement);
        }
        if !body.ends_with(';') {
            body.push(';');
        }

        let kernel_name = format!("{}_{}_{}", self.base_name, dtype_key, ndim);
        // 注意：所有 {、} 在 format! 中都需要转义为 {{、}}，否则会被当作占位符
        let kernel_src = format!(
            r#"
        extern "C" __global__ void {}({}) {{
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements) return;
            {}
            {}
        }}
        "#,
            kernel_name, kernel_args, offset_calc, body
        );

        let kernel = RawKernel::from_source(&kernel_src, &kernel_name, &self.ctx)?;
        let kernel = Arc::new(kernel);
        self.kernels.insert(key, kernel.clone());
        Ok(kernel)
    }
    pub fn launch(&mut self, out: &mut Tensor, inputs: Vec<&Tensor>) -> Result<()> {
        let stream = get_stream()?;
        let stream = stream.inner();
        // 检查所有张量是否连续且位于 GPU
        for inp in &inputs {
            if inp.device() != Device::Cuda(stream.context().ordinal()) {
                bail!("All tensors must be on the same GPU device");
            }
        }
        if out.device() != Device::Cuda(stream.context().ordinal()) {
            bail!("Output tensor must be on the same GPU device");
        }

        let shape = out.shape().to_vec();
        let ndim = shape.len();
        let total = out.size();

        if total == 0 {
            return Ok(());
        }

        // 收集 dtype 用于编译内核
        let mut dtypes = Vec::with_capacity(1 + inputs.len());
        dtypes.push(out.dtype());
        for inp in &inputs {
            dtypes.push(inp.dtype());
        }

        let kernel = self.get_or_compile_kernel(&dtypes, ndim, &shape)?;

        // 先复制输出 strides（只读操作）
        let out_strides = out.strides().to_vec();
        // 再获取可变切片
        let out_slice = out.as_gpu_slice_mut().context("Output not on GPU")?;
        let out_strides_dev = stream
            .clone_htod(&out_strides)
            .context("Failed to copy out_strides to GPU")?;

        // 输入切片和步长
        let mut input_slices = Vec::new();
        let mut input_strides_devs = Vec::new();
        for inp in inputs {
            let slice = inp.as_gpu_slice().context("Input not on GPU")?;
            let strides = inp.strides().to_vec();
            let strides_dev = stream
                .clone_htod(&strides)
                .context("Failed to copy input strides to GPU")?;
            input_slices.push(slice);
            input_strides_devs.push(strides_dev);
        }
        // 拷贝形状
        let shape_dev = stream
            .clone_htod(&shape)
            .context("Failed to copy shape to GPU")?;
        // 启动内核
        let mut builder = kernel.launch_builder(stream);
        builder.arg(out_slice);
        builder.arg(&out_strides_dev);
        for (slice, strides_dev) in input_slices.iter().zip(input_strides_devs.iter()) {
            builder.arg(*slice);
            builder.arg(strides_dev);
        }
        let ndim_i32 = ndim as i32;
        builder.arg(&shape_dev);
        builder.arg(&ndim_i32);
        builder.arg(&total);

        let block = 256;
        let grid = (total + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            builder
                .launch(cfg)
                .context("Failed to launch CUDA kernel")?;
        }
        stream.synchronize();
        Ok(())
    }

    /// 启动内核，使用实现了 TensorViewOps 的视图（支持广播步长）
    pub fn launch_views<V: TensorViewOps>(&mut self, out: &mut V, inputs: Vec<&V>) -> Result<()> {
        use crate::cuda::get_stream;
        use anyhow::Context;

        let stream = get_stream()?;
        let stream = stream.inner();

        let shape = out.shape().to_vec();
        let ndim = shape.len();
        let total = out.size();
        if total == 0 {
            return Ok(());
        }

        // 收集 dtype 用于编译内核
        let mut dtypes = vec![out.dtype()];
        for inp in &inputs {
            dtypes.push(inp.dtype());
        }

        let kernel = self.get_or_compile_kernel(&dtypes, ndim, &shape)?;

        // 输出：CudaViewMut<u8> 和 strides
        let mut out_slice = unsafe { out.as_gpu_view_mut() };
        let out_strides = out.strides().to_vec();
        let out_strides_dev = stream
            .clone_htod(&out_strides)
            .context("Failed to copy out_strides to GPU")?;

        // 输入：CudaView<u8> 和 strides
        let mut input_slices = Vec::with_capacity(inputs.len());
        let mut input_strides_devs = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let slice = unsafe { inp.as_gpu_view() };
            let strides = inp.strides().to_vec();
            let strides_dev = stream
                .clone_htod(&strides)
                .context("Failed to copy input strides to GPU")?;
            input_slices.push(slice);
            input_strides_devs.push(strides_dev);
        }

        let shape_dev = stream
            .clone_htod(&shape)
            .context("Failed to copy shape to GPU")?;

        let mut builder = kernel.launch_builder(stream);
        // 注意：传递引用，满足 PushKernelArg 对 &mut CudaViewMut 和 &CudaView 的实现
        builder.arg(&mut out_slice);
        builder.arg(&out_strides_dev);
        for (slice, strides_dev) in input_slices.iter().zip(input_strides_devs.iter()) {
            builder.arg(slice); // &CudaView<u8>
            builder.arg(strides_dev);
        }
        builder.arg(&shape_dev);
        let ndim_i32 = ndim as i32;
        builder.arg(&ndim_i32);
        builder.arg(&total);

        let block = 256;
        let grid = (total + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            builder.launch(cfg).context("Failed to launch kernel")?;
        }
        stream.synchronize();
        Ok(())
    }
}
