// ndrs/src/backend/cuda/elementwise_kernel.rs
use crate::backend::cuda::RawKernel;
use crate::cuda::get_stream;
use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};
use crate::view::TensorViewOps;
use anyhow::{Context, Result, bail};
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use log::debug;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;

/// 简化的张量参数元数据（仅用于内核代码生成，不含形状/步长/偏移）
struct TensorParam {
    dtype: String, // "float" 或 "int"
    is_output: bool,
}

pub struct ElementwiseKernel {
    base_name: String,
    in_var_names: Vec<String>,
    out_var_name: String,
    in_placeholders: Vec<String>,
    out_placeholder: String,
    raw_statements: Vec<String>, // 存储原始语句，保留类型声明
    ctx: Arc<CudaContext>,
    kernels: HashMap<String, Arc<RawKernel>>,
}

impl ElementwiseKernel {
    pub fn from_expression(
        params_str: &str,
        expr: &str,
        name: &str,
        ctx: Arc<CudaContext>,
    ) -> Result<Self> {
        // 解析参数列表
        let mut all_params = Vec::new();
        for part in params_str.split(',').map(|s| s.trim()) {
            if part.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = part.split_whitespace().collect();
            if tokens.len() != 2 {
                bail!("Invalid param: '{}'", part);
            }
            all_params.push((tokens[0].to_string(), tokens[1].to_string()));
        }
        debug!("All parameters: {:?}", all_params);

        // 分割原始语句
        let raw_statements: Vec<String> = expr
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if raw_statements.is_empty() {
            bail!("Empty expression");
        }

        // 查找输出变量：最后一个没有类型声明的赋值语句的左侧
        let mut output_var = None;
        let mut output_placeholder = None;
        let mut remaining_params = all_params.clone();

        // 从后向前扫描，找到第一个非类型声明语句
        for stmt in raw_statements.iter().rev() {
            // 判断是否为类型声明语句（以大写字母开头，后面有空格）
            let is_decl = stmt
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
                && stmt.contains(' ');
            if !is_decl {
                // 普通赋值，提取左侧变量名
                let eq_pos = stmt
                    .find('=')
                    .ok_or_else(|| anyhow::anyhow!("Missing '=' in statement"))?;
                let lhs = stmt[0..eq_pos].trim();
                if let Some(pos) = remaining_params.iter().position(|(_, v)| v == lhs) {
                    output_var = Some(lhs.to_string());
                    output_placeholder = Some(remaining_params[pos].0.clone());
                    remaining_params.remove(pos);
                    break;
                } else {
                    bail!("Output variable '{}' not found in parameters", lhs);
                }
            }
        }

        let output_var = output_var.ok_or_else(|| anyhow::anyhow!("No output assignment found"))?;
        let output_placeholder = output_placeholder.unwrap();

        // 剩余的都是输入
        let in_var_names: Vec<String> = remaining_params.iter().map(|(_, v)| v.clone()).collect();
        let in_placeholders: Vec<String> =
            remaining_params.iter().map(|(p, _)| p.clone()).collect();

        debug!("Inputs: {:?}", in_var_names);
        debug!("Output: {}", output_var);
        debug!("Raw statements: {:?}", raw_statements);

        Ok(ElementwiseKernel {
            base_name: name.to_string(),
            in_var_names,
            out_var_name: output_var,
            in_placeholders: in_placeholders,
            out_placeholder: output_placeholder,
            raw_statements,
            ctx,
            kernels: HashMap::new(),
        })
    }

    fn generate_kernel_source(
        &self,
        tensors: &HashMap<String, TensorParam>,
        kernel_name: &str,
    ) -> Result<String> {
        // 构建类型占位符到实际 C 类型的映射
        let mut dtype_map = HashMap::new();
        for (ph, var) in self.in_placeholders.iter().zip(&self.in_var_names) {
            let param = tensors.get(var).unwrap();
            dtype_map.insert(ph.clone(), param.dtype.clone());
        }
        {
            let param = tensors.get(&self.out_var_name).unwrap();
            dtype_map.insert(self.out_placeholder.clone(), param.dtype.clone());
        }

        let all_vars: Vec<String> = self
            .in_var_names
            .iter()
            .cloned()
            .chain(std::iter::once(self.out_var_name.clone()))
            .collect();

        // 收集局部变量（从声明语句中提取）
        let mut local_vars = HashMap::new(); // var_name -> c_type
        let decl_re = Regex::new(r"^([A-Z]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=").unwrap();
        for stmt in &self.raw_statements {
            if let Some(caps) = decl_re.captures(stmt) {
                let placeholder = caps[1].to_string();
                let var = caps[2].to_string();
                if all_vars.contains(&var) {
                    bail!("Variable {} conflicts with tensor name", var);
                }
                let c_type = match dtype_map.get(&placeholder) {
                    Some(dt) if dt == "float" => "float",
                    Some(dt) if dt == "int" => "int",
                    _ => bail!("Unknown dtype for placeholder {}", placeholder),
                };
                local_vars.insert(var, c_type.to_string());
            }
        }
        debug!("Local variables: {:?}", local_vars);

        // 构建内核参数列表
        let mut kernel_args = Vec::new();
        for var in &all_vars {
            let param = tensors.get(var).unwrap();
            let qual = if param.is_output { "" } else { "const " };
            kernel_args.push(format!("{}unsigned char* data_{}", qual, var));
            kernel_args.push(format!("const size_t offset_{}", var));
            kernel_args.push(format!("const size_t* strides_{}", var));
        }
        kernel_args.push(
            "const size_t* shape, const size_t ndim, const size_t total_elements".to_string(),
        );
        let kernel_args_str = kernel_args.join(",\n    ");

        // 生成偏移计算代码
        let mut offset_calc = String::new();
        for var in &all_vars {
            offset_calc.push_str(&format!("    size_t off_{} = 0;\n", var));
        }
        offset_calc.push_str("    size_t temp = idx;\n");
        offset_calc.push_str("    for (int d = (int)ndim - 1; d >= 0; --d) {\n");
        offset_calc.push_str("        size_t i = temp % shape[d];\n");
        offset_calc.push_str("        temp /= shape[d];\n");
        for var in &all_vars {
            offset_calc.push_str(&format!("        off_{} += i * strides_{}[d];\n", var, var));
        }
        offset_calc.push_str("    }\n");

        // 转换语句
        let mut transformed = Vec::new();
        let assign_re = Regex::new(r"^([A-Z]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$").unwrap();
        for stmt in &self.raw_statements {
            if let Some(caps) = assign_re.captures(stmt) {
                // 声明语句: TYPE var = expr
                let placeholder = caps[1].to_string();
                let var = caps[2].to_string();
                let rhs = caps[3].to_string();
                let c_type = local_vars.get(&var).unwrap();
                let mut rhs_transformed = rhs.to_string();
                // 替换张量变量
                for tv in &all_vars {
                    let param = tensors.get(tv).unwrap();
                    let access = format!(
                        "*((const {}*)(data_{} + offset_{} + off_{}))",
                        param.dtype, tv, tv, tv
                    );
                    rhs_transformed = replace_whole_word(&rhs_transformed, tv, &access);
                }
                // 替换局部变量（已经声明的）
                for (lv, _) in &local_vars {
                    if lv != &var {
                        rhs_transformed = replace_whole_word(&rhs_transformed, lv, lv);
                    }
                }
                transformed.push(format!("    {} {} = {};", c_type, var, rhs_transformed));
            } else {
                // 普通赋值语句
                let eq_idx = stmt
                    .find('=')
                    .ok_or_else(|| anyhow::anyhow!("Invalid statement: missing '='"))?;
                let lhs = stmt[0..eq_idx].trim();
                let rhs = stmt[eq_idx + 1..].trim();
                let lhs_access = if all_vars.contains(&lhs.to_string()) {
                    let param = tensors.get(lhs).unwrap();
                    format!(
                        "*(({}*)(data_{} + offset_{} + off_{}))",
                        param.dtype, lhs, lhs, lhs
                    )
                } else if local_vars.contains_key(lhs) {
                    lhs.to_string()
                } else {
                    bail!("Unknown left-hand side: {}", lhs);
                };
                let mut rhs_transformed = rhs.to_string();
                for tv in &all_vars {
                    let param = tensors.get(tv).unwrap();
                    let access = format!(
                        "*((const {}*)(data_{} + offset_{} + off_{}))",
                        param.dtype, tv, tv, tv
                    );
                    rhs_transformed = replace_whole_word(&rhs_transformed, tv, &access);
                }
                for (lv, _) in &local_vars {
                    rhs_transformed = replace_whole_word(&rhs_transformed, lv, lv);
                }
                transformed.push(format!("    {} = {};", lhs_access, rhs_transformed));
            }
        }
        let statements_code = transformed.join("\n");

        let kernel_src = format!(
            r#"
extern "C" __global__ void {kernel_name}(
    {kernel_args_str}
) {{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
{offset_calc}{statements_code}
}}
"#,
            kernel_name = kernel_name,
            kernel_args_str = kernel_args_str,
            offset_calc = offset_calc,
            statements_code = statements_code,
        );
        Ok(kernel_src)
    }

    fn get_or_compile_kernel(
        &mut self,
        tensors: &HashMap<String, TensorParam>,
        ndim: usize,
    ) -> Result<Arc<RawKernel>> {
        let mut key = self.base_name.clone();
        for var in &self.in_var_names {
            let param = tensors.get(var).unwrap();
            key.push_str(&format!("_{}_{}", var, param.dtype));
        }
        {
            let param = tensors.get(&self.out_var_name).unwrap();
            key.push_str(&format!("_{}_{}", self.out_var_name, param.dtype));
        }
        key.push_str(&format!("_ndim{}", ndim));
        if let Some(k) = self.kernels.get(&key) {
            return Ok(k.clone());
        }
        let kernel_name = format!("{}_{}", self.base_name, key);
        let src = self.generate_kernel_source(tensors, &kernel_name)?;
        debug!("Generated kernel source:\n{}", src);
        let kernel = RawKernel::from_source(&src, &kernel_name, &self.ctx)?;
        let kernel = Arc::new(kernel);
        self.kernels.insert(key, kernel.clone());
        Ok(kernel)
    }

    pub fn launch_views<V: TensorViewOps>(
        &mut self,
        output: &mut V,
        inputs: Vec<&V>,

        stream: Option<&crate::cuda::Stream>,
    ) -> Result<()> {
        if inputs.len() != self.in_var_names.len() {
            bail!("Input count mismatch");
        }
        let stream = match stream {
            Some(s) => s.inner().clone(),
            None => get_stream()?.inner().clone(),
        };
        let shape = output.shape().to_vec();
        let ndim = shape.len();
        let total = output.size();
        if total == 0 {
            return Ok(());
        }

        // 构建类型映射
        let mut tensors = HashMap::new();
        for (i, var) in self.in_var_names.iter().enumerate() {
            let dtype_str = match inputs[i].dtype() {
                DTYPE_FLOAT32 => "float",
                DTYPE_INT32 => "int",
                _ => bail!("Unsupported dtype"),
            };
            tensors.insert(
                var.clone(),
                TensorParam {
                    dtype: dtype_str.to_string(),
                    is_output: false,
                },
            );
        }
        tensors.insert(
            self.out_var_name.clone(),
            TensorParam {
                dtype: match output.dtype() {
                    DTYPE_FLOAT32 => "float".to_string(),
                    DTYPE_INT32 => "int".to_string(),
                    _ => bail!("Unsupported dtype"),
                },
                is_output: true,
            },
        );

        let kernel = self.get_or_compile_kernel(&tensors, ndim)?;

        let shape_dev = stream.clone_htod(&shape).context("Failed to copy shape")?;

        // 准备输入参数：先获取每个输入的切片（不可变引用）、偏移量、步长设备指针
        // 注意：偏移量是直接值，步长设备指针需要复制步长到设备
        let mut input_slices = Vec::with_capacity(inputs.len());
        let mut input_offsets = Vec::with_capacity(inputs.len());
        let mut input_strides_devs = Vec::with_capacity(inputs.len());

        for inp in &inputs {
            // 先获取切片（不可变借用），但 slices 会保留引用，需要确保在 builder 使用前保持有效
            let slice = unsafe { inp.as_gpu_slice() };
            input_slices.push(slice);
            let offset = inp.offset();
            input_offsets.push(offset);
            let strides_vec = inp.strides().to_vec();
            let strides_dev = stream
                .clone_htod(&strides_vec)
                .context("Failed to copy input strides")?;
            input_strides_devs.push(strides_dev);
        }

        // 准备输出参数：先获取偏移、步长设备指针（不可变操作），再获取切片（可变引用）
        let out_offset = *output.offset();
        let out_strides_vec = output.strides().to_vec();
        let out_strides_dev = stream
            .clone_htod(&out_strides_vec)
            .context("Failed to copy out strides")?;
        // 此刻对 output 的不可变借用已结束（因为 out_offset 和 out_strides_vec 不保留引用）
        let out_slice = unsafe { output.as_gpu_slice_mut() };

        let mut builder = kernel.launch_builder(&stream);

        // 输入参数：按变量名顺序
        for i in 0..self.in_var_names.len() {
            builder.arg(input_slices[i]); // &CudaSlice<u8>
            builder.arg(input_offsets[i]); // &usize
            builder.arg(&input_strides_devs[i]); // &CudaSlice<usize>
        }
        // 输出参数
        builder.arg(out_slice); // &mut CudaSlice<u8> 或 &CudaSlice<u8>
        builder.arg(&out_offset); // &usize
        builder.arg(&out_strides_dev); // &CudaSlice<usize>
        // 全局参数
        builder.arg(&shape_dev);
        builder.arg(&ndim);
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

fn replace_whole_word(s: &str, old: &str, new: &str) -> String {
    let mut res = String::new();
    let ob = old.as_bytes();
    let sb = s.as_bytes();
    let n = sb.len();
    let mut i = 0;
    while i < n {
        if i + ob.len() <= n && &sb[i..i + ob.len()] == ob {
            let prev_ok = i == 0 || !is_ident_char(sb[i - 1] as char);
            let next_ok = i + ob.len() == n || !is_ident_char(sb[i + ob.len()] as char);
            if prev_ok && next_ok {
                res.push_str(new);
                i += ob.len();
                continue;
            }
        }
        res.push(sb[i] as char);
        i += 1;
    }
    res
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::is_available;

    #[test]
    fn test_multi_statement_compilation() {
        if !is_available() {
            return;
        }
        let stream = get_stream().context("Failed to get CUDA stream").unwrap();
        let ctx = stream.inner().context().clone();
        let mut kernel = ElementwiseKernel::from_expression(
            "X x, Y y, Z z",
            "X a = x + 1; Y b = y * 2; z = a + b",
            "test",
            ctx,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "x".to_string(),
            TensorParam {
                dtype: "float".to_string(),
                is_output: false,
            },
        );
        tensors.insert(
            "y".to_string(),
            TensorParam {
                dtype: "float".to_string(),
                is_output: false,
            },
        );
        tensors.insert(
            "z".to_string(),
            TensorParam {
                dtype: "float".to_string(),
                is_output: true,
            },
        );
        let _ = kernel.get_or_compile_kernel(&tensors, 2).unwrap();
    }
}
