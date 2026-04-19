use crate::device::Device;
use crate::kernel::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub type DType = u32;

pub const DTYPE_FLOAT32: DType = 1;
pub const DTYPE_INT32: DType = 2;

#[derive(Clone, Copy)]
pub struct TypeInfo {
    pub size: usize,
    pub name: &'static str,
}

pub type BinaryOp = Arc<
    dyn Fn(
            *const u8,
            *const usize, // a, a_strides
            *const u8,
            *const usize, // b, b_strides
            *mut u8,
            *const usize, // c, c_strides
            *const usize, // shape
            usize,        // ndim
            usize,        // total_elements
            Device,
            Option<*mut std::ffi::c_void>,
        ) + Send
        + Sync,
>;

struct TypeRegistryInner {
    info: HashMap<DType, TypeInfo>,
    add_op: HashMap<DType, BinaryOp>,
}

impl TypeRegistryInner {
    fn new() -> Self {
        TypeRegistryInner {
            info: HashMap::new(),
            add_op: HashMap::new(),
        }
    }
}

pub struct TypeRegistry(RwLock<TypeRegistryInner>);

impl TypeRegistry {
    fn new() -> Self {
        TypeRegistry(RwLock::new(TypeRegistryInner::new()))
    }

    pub fn register(&self, dtype: DType, info: TypeInfo, add_op: BinaryOp) {
        let mut inner = self.0.write().unwrap();
        inner.info.insert(dtype, info);
        inner.add_op.insert(dtype, add_op);
    }

    pub fn get_info(&self, dtype: DType) -> Option<TypeInfo> {
        self.0.read().unwrap().info.get(&dtype).copied()
    }

    pub fn get_add_op(&self, dtype: DType) -> Option<BinaryOp> {
        self.0.read().unwrap().add_op.get(&dtype).cloned()
    }
}

pub static TYPE_REGISTRY: Lazy<TypeRegistry> = Lazy::new(|| {
    let reg = TypeRegistry::new();

    // float32
    reg.register(
        DTYPE_FLOAT32,
        TypeInfo {
            size: 4,
            name: "float32",
        },
        Arc::new(
            |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, stream| {
                let a_ptr = a as *const f32;
                let b_ptr = b as *const f32;
                let c_ptr = c as *mut f32;
                match dev {
                    Device::CPU => unsafe {
                        cpu_strided_add_f32(
                            a_ptr,
                            a_strides,
                            b_ptr,
                            b_strides,
                            c_ptr,
                            c_strides,
                            shape,
                            ndim as i32,
                            n,
                        )
                    },
                    Device::GPU(_) => unsafe {
                        gpu_strided_add_f32(
                            a_ptr,
                            a_strides,
                            b_ptr,
                            b_strides,
                            c_ptr,
                            c_strides,
                            shape,
                            ndim as i32,
                            n,
                            stream.unwrap(),
                        );
                    },
                }
            },
        ),
    );

    // int32
    reg.register(
        DTYPE_FLOAT32,
        TypeInfo {
            size: 4,
            name: "float32",
        },
        Arc::new(
            |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, stream| {
                let a_ptr = a as *const i32;
                let b_ptr = b as *const i32;
                let c_ptr = c as *mut i32;
                match dev {
                    Device::CPU => unsafe {
                        cpu_strided_add_i32(
                            a_ptr,
                            a_strides,
                            b_ptr,
                            b_strides,
                            c_ptr,
                            c_strides,
                            shape,
                            ndim as i32,
                            n,
                        )
                    },
                    Device::GPU(_) => unsafe {
                        gpu_strided_add_i32(
                            a_ptr,
                            a_strides,
                            b_ptr,
                            b_strides,
                            c_ptr,
                            c_strides,
                            shape,
                            ndim as i32,
                            n,
                            stream.unwrap(),
                        );
                    },
                }
            },
        ),
    );

    reg
});

pub fn register_dtype(dtype: DType, info: TypeInfo) {
    TYPE_REGISTRY.0.write().unwrap().info.insert(dtype, info);
}

pub fn register_add_op(dtype: DType, op: BinaryOp) {
    TYPE_REGISTRY.0.write().unwrap().add_op.insert(dtype, op);
}

pub fn get_dtype_info(dtype: DType) -> Option<TypeInfo> {
    TYPE_REGISTRY.get_info(dtype)
}

pub fn get_add_op(dtype: DType) -> Option<BinaryOp> {
    TYPE_REGISTRY.get_add_op(dtype)
}

pub trait DTypeMapping {
    const DTYPE: DType;
}

impl DTypeMapping for f32 {
    const DTYPE: DType = DTYPE_FLOAT32;
}

impl DTypeMapping for i32 {
    const DTYPE: DType = DTYPE_INT32;
}
