//! 切片宏实现

#[macro_export]
macro_rules! s {
    // 基规则：解析完成
    (@parse [$($elems:expr),*] ) => { $crate::view::SliceInfo::new(vec![$($elems),*]) };

    // ==================== 单独常数（整数） ====================
    (@parse [$($elems:expr),*] $num:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Index($num)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $num:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Index($num)] )
    };

    // ==================== a..=b（包含右边界） ====================
    (@parse [$($elems:expr),*] $s:literal ..= $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive($s, $e)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal ..= $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive($s, $e)] )
    };
    (@parse [$($elems:expr),*] - $s:literal ..= $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive(-$s, $e)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal ..= $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive(-$s, $e)] )
    };
    (@parse [$($elems:expr),*] $s:literal ..= - $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive($s, -$e)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal ..= - $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive($s, -$e)] )
    };
    (@parse [$($elems:expr),*] - $s:literal ..= - $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive(-$s, -$e)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal ..= - $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::RangeInclusive(-$s, -$e)] )
    };

    // ==================== a..b（普通范围，步长1） ====================
    (@parse [$($elems:expr),*] $s:literal .. $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, 1)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, 1)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, 1)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, 1)] )
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, 1)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, 1)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, 1)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, 1)] )
    };

    // ==================== a..b;s（带步长） ====================
    (@parse [$($elems:expr),*] $s:literal .. $e:literal ; $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, $step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. $e:literal ; $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, $step)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal ; $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, $step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal ; $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, $step)] )
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal ; $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, $step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal ; $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, $step)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal ; $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, $step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal ; $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, $step)] )
    };
    // step 为负
    (@parse [$($elems:expr),*] $s:literal .. $e:literal ; - $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, -$step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. $e:literal ; - $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, $e, -$step)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal ; - $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, -$step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. $e:literal ; - $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, $e, -$step)] )
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal ; - $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, -$step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. - $e:literal ; - $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range($s, -$e, -$step)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal ; - $step:literal , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, -$step)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. - $e:literal ; - $step:literal ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::Range(-$s, -$e, -$step)] )
    };

    // ==================== a..（只有起点） ====================
    (@parse [$($elems:expr),*] $s:literal .. , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::From($s)] $($rest)*)
    };
    (@parse [$($elems:expr),*] $s:literal .. ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::From($s)] )
    };
    (@parse [$($elems:expr),*] - $s:literal .. , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::From(-$s)] $($rest)*)
    };
    (@parse [$($elems:expr),*] - $s:literal .. ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::From(-$s)] )
    };

    // ==================== ..（全范围） ====================
    (@parse [$($elems:expr),*] .. , $($rest:tt)*) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::All] $($rest)*)
    };
    (@parse [$($elems:expr),*] .. ) => {
        s!(@parse [$($elems,)* $crate::view::SliceArg::All] )
    };

    // 入口
    ($($rest:tt)*) => {
        s!(@parse [] $($rest)*)
    };
}

#[macro_export]
macro_rules! tensor_data {
    // 显式指定类型后缀（i32 或 f32）
    ($data:tt ; $t:ident) => {{
        let (strings, shape): (Vec<&str>, Vec<usize>) = $crate::tensor_data!(@flatten $data);
        let dtype = stringify!($t);
        (strings, shape, dtype)
    }};

    // 无类型后缀，dtype = "none"
    ($data:tt) => {{
        let (strings, shape): (Vec<&str>, Vec<usize>) = $crate::tensor_data!(@flatten $data);
        let dtype = "none";
        (strings, shape, dtype)
    }};

    // ========== 递归扁平化辅助规则 ==========
    // 标量
    (@flatten $x:literal) => {
        (vec![stringify!($x)], Vec::<usize>::new())
    };
    // 空数组
    (@flatten []) => {
        (Vec::<&str>::new(), vec![0usize])
    };
    // 非空数组（递归展开）
    (@flatten [$($inner:tt),* $(,)?]) => {{
        let mut all_strings: Vec<&str> = Vec::new();
        let mut inner_shape: Option<Vec<usize>> = None;
        let mut count = 0usize;
        $(
            let (inner_strings, shape) = $crate::tensor_data!(@flatten $inner);
            all_strings.extend(inner_strings);
            if let Some(ref s) = inner_shape {
                if shape != *s {
                    panic!("Inconsistent dimensions in tensor_data! macro");
                }
            } else {
                inner_shape = Some(shape);
            }
            count += 1;
        )*
        let inner_shape = inner_shape.unwrap_or_else(Vec::new);
        let mut shape = vec![count];
        shape.extend(inner_shape);
        (all_strings, shape)
    }};
}

#[macro_export]
macro_rules! tensor {
    // 标量
    ($x:literal) => {{
        let (strings, shape, dtype_str) = $crate::tensor_data!($x);
        let dtype_hint = if dtype_str == "none" { None } else { Some(dtype_str) };
        $crate::Tensor::from_strings(&strings, &shape, dtype_hint)
            .expect("Failed to create tensor from data")
    }};

    // 方括号数组字面量：tensor![[1, 2], [3, 4]]
    ([$($data:tt)*]) => {{
        // $data 捕获内部 token，重新包装成 [$($data)*] 传给 tensor_data!
        let (strings, shape, dtype_str) = $crate::tensor_data!([$($data)*]);
        let dtype_hint = if dtype_str == "none" { None } else { Some(dtype_str) };
        $crate::Tensor::from_strings(&strings, &shape, dtype_hint)
            .expect("Failed to create tensor from data")
    }};

    // 圆括号数组字面量：tensor!([1, 2, 3])
    (($($data:tt)*)) => {{
        let (strings, shape, dtype_str) = $crate::tensor_data!([$($data)*]);
        let dtype_hint = if dtype_str == "none" { None } else { Some(dtype_str) };
        $crate::Tensor::from_strings(&strings, &shape, dtype_hint)
            .expect("Failed to create tensor from data")
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};
    #[test]
    fn test_tensor_data_macro() {
        // 空数组
        let (data, shape, dtype) = tensor_data!([]);
        assert_eq!(data, Vec::<&str>::new());
        assert_eq!(shape, vec![0]);
        assert_eq!(dtype, "none");

        // 嵌套空数组
        let (data, shape, dtype) = tensor_data!([[], []]);
        assert_eq!(data, Vec::<&str>::new());
        assert_eq!(shape, vec![2, 0]);
        assert_eq!(dtype, "none");

        // 标量
        let (data, shape, dtype) = tensor_data!(42);
        assert_eq!(data, vec!["42"]);
        assert_eq!(shape, Vec::<usize>::new());
        assert_eq!(dtype, "none");

        // 一维整数
        let (data, shape, dtype) = tensor_data!([1, 2, 3]);
        assert_eq!(data, vec!["1", "2", "3"]);
        assert_eq!(shape, vec![3]);
        assert_eq!(dtype, "none");

        // 二维浮点数
        let (data, shape, dtype) = tensor_data!([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(data, vec!["1.0", "2.0", "3.0", "4.0"]);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(dtype, "none");

        // 混合自动推断为 f32
        let (data, shape, dtype) = tensor_data!([[[1, 2.0], [3, 4]], [[5.0, 6.0], [7, 8]]]);
        assert_eq!(data, vec!["1", "2.0", "3", "4", "5.0", "6.0", "7", "8"]);
        assert_eq!(shape, vec![2, 2, 2]);
        assert_eq!(dtype, "none");

        // 显式指定 f32
        let (data, shape, dtype) = tensor_data!([1, 2, 3]; i32);
        assert_eq!(data, vec!["1", "2", "3"]);
        assert_eq!(shape, vec![3]);
        assert_eq!(dtype, "i32");

        // 科学计数法自动 f32
        let (data, shape, dtype) = tensor_data!([1e-3, 2.5]; f32);
        assert_eq!(data, vec!["1e-3", "2.5"]);
        assert_eq!(shape, vec![2]);
        assert_eq!(dtype, "f32");
    }

    #[test]
    fn test_tensor_macro() {
        let t = tensor!([[1, 2], [3, 4]]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DTYPE_INT32);
        assert_eq!(t.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);

        let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let t = tensor!([[1, 2.0], [3, 4]]);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let t = tensor!([]);
        assert_eq!(t.shape(), &[0]);
        assert_eq!(t.size(), 0);
    }
}
