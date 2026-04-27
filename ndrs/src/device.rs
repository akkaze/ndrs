use anyhow::{Result, bail};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    pub fn as_cuda_index(&self) -> Option<usize> {
        match self {
            Device::Cuda(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        if s == "cpu" {
            Ok(Device::Cpu)
        } else if let Some(stripped) = s.strip_prefix("cuda:") {
            let idx = stripped.parse::<usize>()?;
            Ok(Device::Cuda(idx))
        } else {
            bail!("Unknown device specifier: {}", s)
        }
    }
}
