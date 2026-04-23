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
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "cpu" {
            Ok(Device::Cpu)
        } else if let Some(stripped) = s.strip_prefix("cuda:") {
            let idx = stripped
                .parse::<usize>()
                .map_err(|_| format!("Invalid cuda index: {}", stripped))?;
            Ok(Device::Cuda(idx))
        } else {
            Err(format!("Unknown device specifier: {}", s))
        }
    }
}
