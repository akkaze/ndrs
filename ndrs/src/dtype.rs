// ndrs/src/dtype.rs
use crate::device::Device;
use crate::kernel::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

pub type DType = u32;

pub const DTYPE_FLOAT32: DType = 1;
pub const DTYPE_INT32: DType = 2;

static NEXT_DTYPE_ID: AtomicU32 = AtomicU32::new(1000);

pub fn allocate_dtype() -> DType {
    NEXT_DTYPE_ID.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone)]
pub struct TypeInfo {
    pub size: usize,
    pub name: String,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOpKind {
    pub fn as_u32(&self) -> u32 {
        match self {
            BinaryOpKind::Add => 0,
            BinaryOpKind::Sub => 1,
            BinaryOpKind::Mul => 2,
            BinaryOpKind::Div => 3,
        }
    }
}

pub type BinaryOpFn = Arc<
    dyn Fn(
            *const u8,
            *const usize,
            *const u8,
            *const usize,
            *mut u8,
            *const usize,
            *const usize,
            usize,
            usize,
            Device,
            Option<*mut std::ffi::c_void>,
        ) -> Result<(), String>
        + Send
        + Sync,
>;

struct TypeRegistryInner {
    info: HashMap<DType, TypeInfo>,
    binary_ops: HashMap<(DType, BinaryOpKind, Device), BinaryOpFn>,
}

impl TypeRegistryInner {
    fn new() -> Self {
        TypeRegistryInner {
            info: HashMap::new(),
            binary_ops: HashMap::new(),
        }
    }
}

pub struct TypeRegistry(RwLock<TypeRegistryInner>);

impl TypeRegistry {
    fn new() -> Self {
        TypeRegistry(RwLock::new(TypeRegistryInner::new()))
    }

    pub fn register_dtype(&self, dtype: DType, info: TypeInfo) {
        let mut inner = self.0.write().unwrap();
        inner.info.insert(dtype, info);
    }

    pub fn register_binary_op(
        &self,
        dtype: DType,
        kind: BinaryOpKind,
        device: Device,
        op: BinaryOpFn,
    ) -> Result<(), String> {
        let mut inner = self.0.write().unwrap();
        inner.binary_ops.insert((dtype, kind, device), op);
        Ok(())
    }

    pub fn get_info(&self, dtype: DType) -> Option<TypeInfo> {
        self.0.read().unwrap().info.get(&dtype).cloned()
    }

    pub fn get_binary_op(
        &self,
        dtype: DType,
        kind: BinaryOpKind,
        device: Device,
    ) -> Option<BinaryOpFn> {
        self.0
            .read()
            .unwrap()
            .binary_ops
            .get(&(dtype, kind, device))
            .cloned()
    }
}

pub static TYPE_REGISTRY: Lazy<TypeRegistry> = Lazy::new(|| {
    let reg = TypeRegistry::new();

    // 内置类型信息（使用 String）
    reg.register_dtype(
        DTYPE_FLOAT32,
        TypeInfo {
            size: 4,
            name: "float32".to_string(),
        },
    );
    reg.register_dtype(
        DTYPE_INT32,
        TypeInfo {
            size: 4,
            name: "int32".to_string(),
        },
    );

    // CPU 加法
    let add_f32_cpu: BinaryOpFn = Arc::new(
        |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, _| {
            let a_ptr = a as *const f32;
            let b_ptr = b as *const f32;
            let c_ptr = c as *mut f32;
            unsafe {
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
                );
            }
            Ok(())
        },
    );
    reg.register_binary_op(DTYPE_FLOAT32, BinaryOpKind::Add, Device::Cpu, add_f32_cpu)
        .unwrap();

    // GPU 加法
    let add_f32_gpu: BinaryOpFn = Arc::new(
        |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, stream| {
            let a_ptr = a as *const f32;
            let b_ptr = b as *const f32;
            let c_ptr = c as *mut f32;
            unsafe {
                let err = gpu_strided_add_f32(
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
                if err != 0 {
                    return Err(format!("GPU add failed: {}", err));
                }
            }
            Ok(())
        },
    );
    reg.register_binary_op(
        DTYPE_FLOAT32,
        BinaryOpKind::Add,
        Device::Cuda(0),
        add_f32_gpu,
    )
    .unwrap();

    let add_i32_cpu: BinaryOpFn = Arc::new(
        |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, _| {
            let a_ptr = a as *const i32;
            let b_ptr = b as *const i32;
            let c_ptr = c as *mut i32;
            unsafe {
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
                );
            }
            Ok(())
        },
    );
    reg.register_binary_op(DTYPE_INT32, BinaryOpKind::Add, Device::Cpu, add_i32_cpu)
        .unwrap();

    let add_i32_gpu: BinaryOpFn = Arc::new(
        |a, a_strides, b, b_strides, c, c_strides, shape, ndim, n, dev, stream| {
            let a_ptr = a as *const i32;
            let b_ptr = b as *const i32;
            let c_ptr = c as *mut i32;
            unsafe {
                let err = gpu_strided_add_i32(
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
                if err != 0 {
                    return Err(format!("GPU add failed: {}", err));
                }
            }
            Ok(())
        },
    );
    reg.register_binary_op(DTYPE_INT32, BinaryOpKind::Add, Device::Cuda(0), add_i32_gpu)
        .unwrap();

    reg
});

// 公共 API
pub fn register_dtype(dtype: DType, info: TypeInfo) {
    TYPE_REGISTRY.register_dtype(dtype, info);
}

pub fn register_binary_op(
    dtype: DType,
    kind: BinaryOpKind,
    device: Device,
    op: BinaryOpFn,
) -> Result<(), String> {
    TYPE_REGISTRY.register_binary_op(dtype, kind, device, op)
}

pub fn get_dtype_info(dtype: DType) -> Option<TypeInfo> {
    TYPE_REGISTRY.get_info(dtype)
}

pub fn get_binary_op(dtype: DType, kind: BinaryOpKind, device: Device) -> Option<BinaryOpFn> {
    TYPE_REGISTRY.get_binary_op(dtype, kind, device)
}

// 为了方便旧代码，保留 get_add_op 但需要 device 参数
pub fn get_add_op(dtype: DType, device: Device) -> Option<BinaryOpFn> {
    get_binary_op(dtype, BinaryOpKind::Add, device)
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
