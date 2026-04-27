use anyhow::{Context, Result, anyhow, bail};
/// Tensor I/O: Loading and saving from/to NumPy .npy files using npyz.
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

use npyz::WriterBuilder;
use npyz::{DType, NpyFile, TypeStr, WriteOptions};

use crate::device::Device;
use crate::dtype::{DTYPE_FLOAT32, DTYPE_INT32};
use crate::tensor::{RcTensor, Tensor};

/// Load a Tensor from a .npy file.
pub fn load_npy<P: AsRef<Path>>(path: P) -> anyhow::Result<Tensor> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let npy = NpyFile::new(reader)?;

    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();

    let dtype_str = match npy.dtype() {
        DType::Plain(ts) => ts.to_string(),
        _ => bail!("Only plain dtypes are supported"),
    };

    if dtype_str == "<f4" || dtype_str == ">f4" || dtype_str == "|f4" {
        let data: Vec<f32> = npy.into_vec()?;
        Tensor::from_vec(data, shape, Device::Cpu)
    } else if dtype_str == "<i4" || dtype_str == ">i4" || dtype_str == "|i4" {
        let data: Vec<i32> = npy.into_vec()?;
        Tensor::from_vec(data, shape, Device::Cpu)
    } else {
        bail!("Unsupported dtype: {}", dtype_str)
    }
}

/// Save a Tensor to a .npy file. The tensor must be on CPU and contiguous.
pub fn save_npy<P: AsRef<Path>>(tensor: &Tensor, path: P) -> anyhow::Result<()> {
    if tensor.device() != Device::Cpu {
        bail!("save_npy only supports CPU tensors");
    }
    if !tensor.is_contiguous() {
        bail!("save_npy requires contiguous tensor");
    }

    let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    match tensor.dtype() {
        DTYPE_FLOAT32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_bytes().unwrap().as_ptr() as *const f32,
                    tensor.size(),
                )
            };
            let type_str = TypeStr::from_str("<f4")?;
            let dtype = DType::Plain(type_str);
            let mut w = WriteOptions::new()
                .dtype(dtype)
                .shape(&shape)
                .writer(writer)
                .begin_nd()?;
            w.extend(slice)?;
            w.finish()?;
        }
        DTYPE_INT32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_bytes().unwrap().as_ptr() as *const i32,
                    tensor.size(),
                )
            };
            let type_str = TypeStr::from_str("<i4")?;
            let dtype = DType::Plain(type_str);
            let mut w = WriteOptions::new()
                .dtype(dtype)
                .shape(&shape)
                .writer(writer)
                .begin_nd()?;
            w.extend(slice)?;
            w.finish()?;
        }
        _ => return bail!("Unsupported dtype: {}", tensor.dtype()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DTYPE_FLOAT32, DTYPE_INT32, tensor};
    use tempfile::NamedTempFile;

    #[test]
    fn test_npy_roundtrip_f32() {
        let tensor = tensor!([[1.0, 2.0], [3.0, 4.0]]);
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_npy(&tensor, path).unwrap();
        let loaded = load_npy(path).unwrap();

        assert_eq!(loaded.shape(), &[2, 2]);
        assert_eq!(loaded.dtype(), DTYPE_FLOAT32);
        assert_eq!(loaded.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_npy_roundtrip_i32() {
        let tensor = tensor!([[-1, 2], [3, -4]]);
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_npy(&tensor, path).unwrap();
        let loaded = load_npy(path).unwrap();

        assert_eq!(loaded.shape(), &[2, 2]);
        assert_eq!(loaded.dtype(), DTYPE_INT32);
        assert_eq!(loaded.to_vec::<i32>().unwrap(), vec![-1, 2, 3, -4]);
    }
}
