//! 切片类型定义，由 s! 宏生成

#[derive(Debug, Clone)]
pub enum SliceArg {
    Index(usize),
    Range(usize, usize, usize),
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