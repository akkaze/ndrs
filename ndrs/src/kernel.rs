use std::ffi::c_void;

extern "C" {
    pub fn cpu_add_f32(a: *const f32, b: *const f32, c: *mut f32, n: usize);
    pub fn gpu_add_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn cpu_add_i32(a: *const i32, b: *const i32, c: *mut i32, n: usize);
    pub fn gpu_add_i32(
        a: *const i32,
        b: *const i32,
        c: *mut i32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;

    pub fn cpu_strided_copy(
        src: *const u8,
        src_offset: usize,
        src_strides: *const usize,
        ndim: i32,
        shape: *const usize,
        dst: *mut u8,
        dst_offset: usize,
        dst_strides: *const usize,
        elem_size: usize,
        total_elements: usize,
    );

    pub fn gpu_strided_copy(
        src: *const u8,
        src_offset: usize,
        src_strides: *const usize,
        ndim: i32,
        shape: *const usize,
        dst: *mut u8,
        dst_offset: usize,
        dst_strides: *const usize,
        elem_size: usize,
        total_elements: usize,
        stream: *mut c_void,
    ) -> i32;

    pub fn cpu_contiguous(
        src: *const u8,
        src_offset: usize,
        src_strides: *const usize,
        ndim: i32,
        shape: *const usize,
        dst: *mut u8,
        elem_size: usize,
        total_elements: usize,
    );

    pub fn gpu_contiguous(
        src: *const u8,
        src_offset: usize,
        src_strides: *const usize,
        ndim: i32,
        shape: *const usize,
        dst: *mut u8,
        elem_size: usize,
        total_elements: usize,
        stream: *mut c_void,
    ) -> i32;
}
