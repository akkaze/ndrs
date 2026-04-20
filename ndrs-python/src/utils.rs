use pyo3::prelude::*;
use pyo3::types::PyList;

/// Flatten a nested Python list into a flat Vec<f32> and record the shape.
pub fn flatten_list_f32(obj: &PyAny, shape: &mut Vec<usize>) -> PyResult<Vec<f32>> {
    if let Ok(list) = obj.extract::<&PyList>() {
        shape.push(list.len());
        let mut result = Vec::new();
        for item in list.iter() {
            result.extend(flatten_list_f32(item, &mut Vec::new())?);
        }
        Ok(result)
    } else {
        let val: f64 = obj.extract()?;
        Ok(vec![val as f32])
    }
}

#[allow(dead_code)]
pub fn flatten_list_i32(obj: &PyAny, shape: &mut Vec<usize>) -> PyResult<Vec<i32>> {
    if let Ok(list) = obj.extract::<&PyList>() {
        shape.push(list.len());
        let mut result = Vec::new();
        for item in list.iter() {
            result.extend(flatten_list_i32(item, &mut Vec::new())?);
        }
        Ok(result)
    } else {
        let val: i64 = obj.extract()?;
        Ok(vec![val as i32])
    }
}