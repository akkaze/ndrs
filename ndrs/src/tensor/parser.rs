use crate::device::Device;
use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32, DType};

/// 去除字符串首尾的双引号（如果存在）
fn strip_quotes(token: &str) -> &str {
    let token = token.trim();
    if token.starts_with('"') && token.ends_with('"') && token.len() >= 2 {
        &token[1..token.len() - 1]
    } else {
        token
    }
}

/// 解析完整的张量字符串，返回 (字符串列表, 形状, 可选的dtype, 设备)
pub(crate) fn parse_full_tensor_string(
    s: &str,
) -> Result<(Vec<&str>, Vec<usize>, Option<DType>, Device), String> {
    let s = s.trim();

    // 分离数据部分和后缀部分（用分号分割）
    let parts: Vec<&str> = s.split(';').map(|p| p.trim()).collect();
    if parts.is_empty() {
        return Err("Empty string".into());
    }

    let data_part = parts[0];
    let (strings, shape) = parse_nested_array(data_part)?;

    let mut dtype: Option<DType> = None;
    let mut device: Device = Device::Cpu; // 默认 CPU

    for part in &parts[1..] {
        if part.is_empty() {
            continue;
        }
        let token = strip_quotes(part);
        // 尝试解析为设备字符串（如 "cpu", "cuda:0"）
        if let Ok(dev) = token.parse::<Device>() {
            device = dev;
            continue;
        }
        // 否则尝试解析为 dtype
        match token {
            "f32" => dtype = Some(DTYPE_FLOAT32),
            "i32" => dtype = Some(DTYPE_INT32),
            _ => {
                return Err(format!(
                    "Unrecognized suffix token: '{}'. Expected dtype (f32/i32) or device (cpu/cuda:N)",
                    token
                ));
            }
        }
    }

    Ok((strings, shape, dtype, device))
}

/// 递归解析嵌套数组字符串，例如 "[[1,2],[3,4]]" -> (vec!["1","2","3","4"], vec![2,2])
/// 内部函数，不处理后缀。
fn parse_nested_array(s: &str) -> Result<(Vec<&str>, Vec<usize>), String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok((vec![], vec![0]));
    }
    // 去除最外层括号
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        // 标量：没有括号
        return Ok((vec![s], vec![]));
    };
    // 空数组
    if inner.is_empty() {
        return Ok((vec![], vec![0]));
    }
    // 分割顶层元素
    let mut elements = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let chars: Vec<char> = inner.chars().collect();
    for i in 0..chars.len() {
        match chars[i] {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                let elem = inner[start..i].trim();
                elements.push(elem);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < inner.len() {
        let elem = inner[start..].trim();
        elements.push(elem);
    }

    let mut all_strings = Vec::new();
    let mut child_shapes = Vec::new();
    for elem in &elements {
        let (strings, shape) = parse_nested_array(elem)?;
        all_strings.extend(strings);
        child_shapes.push(shape);
    }
    if child_shapes.is_empty() {
        return Ok((vec![], vec![0]));
    }
    let first_shape = &child_shapes[0];
    for shape in &child_shapes[1..] {
        if shape != first_shape {
            return Err("Inconsistent dimensions".into());
        }
    }
    let mut shape = vec![elements.len()];
    shape.extend(first_shape);
    Ok((all_strings, shape))
}
