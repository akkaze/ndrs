use ndrs::{register_dtype, register_add_op, TypeInfo, DType, Device};
use std::sync::Arc;

const DTYPE_MY_TYPE: DType = 1000;

#[repr(C)]
#[derive(Clone, Copy)]
struct MyType {
    a: i32,
    b: i32,
}

impl MyType {
    fn add(&self, other: &MyType) -> MyType {
        MyType { a: self.a + other.a, b: self.b + other.b }
    }
}

fn mytype_add_op(a: *const u8, b: *const u8, c: *mut u8, n: usize, _dev: Device, _stream: Option<*mut std::ffi::c_void>) {
    let a_slice = unsafe { std::slice::from_raw_parts(a as *const MyType, n) };
    let b_slice = unsafe { std::slice::from_raw_parts(b as *const MyType, n) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c as *mut MyType, n) };
    for i in 0..n {
        c_slice[i] = a_slice[i].add(&b_slice[i]);
    }
}

fn main() {
    register_dtype(DTYPE_MY_TYPE, TypeInfo { size: std::mem::size_of::<MyType>(), name: "mytype" });
    register_add_op(DTYPE_MY_TYPE, Box::new(mytype_add_op));
    println!("Custom type registered");
}