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
macro_rules! tensor {
    ($($tt:tt)*) => {{
        let s = stringify!($($tt)*);
        $crate::Tensor::from_string_literal(s)
            .expect("Failed to create tensor from string literal")
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};

    #[test]
    fn test_tensor_macro() {
        let t = tensor!([[1, -2], [3, 4]]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DTYPE_INT32);
        assert_eq!(t.to_vec::<i32>().unwrap(), vec![1, -2, 3, 4]);

        let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let t = tensor!([[1, 2.0], [3, -4]]);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, -4.0]);

        let t = tensor!([]);
        assert_eq!(t.shape(), &[0]);
        assert_eq!(t.size(), 0);
    }
}
