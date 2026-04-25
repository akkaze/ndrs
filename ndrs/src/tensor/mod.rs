use crate::dtype::DType;

mod access;
mod construction;
mod conversion;
mod data;
mod handle;
mod handle_ops;
mod io;
mod parser;

// 不再重新导出 DataPtr
use crate::Device;
pub use access::*;
pub use construction::*;
pub use conversion::*;
pub use data::*;
pub use handle::{ArcTensor, RcTensor};
pub use io::{load_npy, save_npy};

#[derive(Debug)]
pub struct Tensor {
    pub(crate) data: DataPtr,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DTYPE_FLOAT32;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.size(), 3);
        assert_eq!(t.dtype(), DTYPE_FLOAT32);
        assert!(t.is_contiguous());
        assert_eq!(t.strides(), &[4]);
    }

    #[test]
    fn test_tensor_bytes() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let bytes = t.as_bytes().unwrap();
        assert_eq!(bytes.len(), 8);
        let values: Vec<f32> =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 2).to_vec() };
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_tensor_into_rc() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let rc = t.into_rc();
        assert_eq!(rc.0.borrow().shape(), &[2]);
    }

    #[test]
    fn test_tensor_into_arc() {
        let t = Tensor::new_cpu_from_f32(vec![1.0, 2.0], vec![2]);
        let arc = t.into_arc();
        assert_eq!(arc.0.lock().borrow().shape(), &[2]);
    }
}
