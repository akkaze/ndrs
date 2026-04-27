/// 广播形状计算
use anyhow::{Context, Result, anyhow, bail};

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result = Vec::new();
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result.push(std::cmp::max(dim1, dim2));
        } else {
            return None;
        }
    }
    result.reverse();
    Some(result)
}

fn compute_broadcast_strides(
    in_shape: &[usize],
    out_shape: &[usize],
    in_strides: &[usize],
) -> Result<Vec<usize>> {
    let in_ndim = in_shape.len();
    let out_ndim = out_shape.len();
    let mut bcast_strides = vec![0; out_ndim];
    // 对齐尾部
    for i in 0..out_ndim {
        let out_dim = out_shape[out_ndim - 1 - i];
        let in_dim = if i < in_ndim {
            in_shape[in_ndim - 1 - i]
        } else {
            1
        };
        let in_stride = if i < in_ndim {
            in_strides[in_ndim - 1 - i]
        } else {
            0
        };
        if in_dim == out_dim {
            bcast_strides[out_ndim - 1 - i] = in_stride;
        } else if in_dim == 1 {
            bcast_strides[out_ndim - 1 - i] = 0;
        } else {
            bail!(
                "Cannot broadcast dimension {}: {} vs {}",
                i,
                in_dim,
                out_dim
            );
        }
    }
    Ok(bcast_strides)
}

#[cfg(test)]
mod tests {
    use super::broadcast_shapes;

    #[test]
    fn test_broadcast_shapes() {
        let a = vec![3, 1];
        let b = vec![1, 4];
        let result = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(result, vec![3, 4]);

        let a = vec![2, 3];
        let b = vec![3];
        let result = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(result, vec![2, 3]);

        let a = vec![2];
        let b = vec![3];
        assert!(broadcast_shapes(&a, &b).is_none());
    }
}
