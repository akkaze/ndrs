//! 切片类型定义，由 s! 宏生成

#[derive(Debug, Clone)]
pub enum SliceArg {
    /// 单个索引（支持负数表示从末尾开始）
    Index(i32),
    /// 范围 [start, end)，步长 step（end 可为负数，表示从末尾开始）
    Range(i32, i32, i32),
    /// 范围 [start, end]，步长 1
    RangeInclusive(i32, i32),
    /// 从 start 到末尾
    From(i32),
    /// 全选
    All,
}

pub struct SliceInfo {
    args: Vec<SliceArg>,
}

impl SliceInfo {
    pub fn new(args: Vec<SliceArg>) -> Self {
        SliceInfo { args }
    }
    pub fn args(&self) -> &[SliceArg] {
        &self.args
    }
}
