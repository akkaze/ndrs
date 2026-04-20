use std::ffi::c_void;

extern "C" {
    pub fn cpu_strided_add_f32(
        a: *const f32,
        a_strides: *const usize,
        b: *const f32,
        b_strides: *const usize,
        c: *mut f32,
        c_strides: *const usize,
        shape: *const usize,
        ndim: i32,
        total_elements: usize,
    );
    pub fn gpu_strided_add_f32(
        a: *const f32,
        a_strides: *const usize,
        b: *const f32,
        b_strides: *const usize,
        c: *mut f32,
        c_strides: *const usize,
        shape: *const usize,
        ndim: i32,
        total_elements: usize,
        stream: *mut c_void,
    ) -> i32;

    pub fn cpu_strided_add_i32(
        a: *const i32,
        a_strides: *const usize,
        b: *const i32,
        b_strides: *const usize,
        c: *mut i32,
        c_strides: *const usize,
        shape: *const usize,
        ndim: i32,
        total_elements: usize,
    );
    pub fn gpu_strided_add_i32(
        a: *const i32,
        a_strides: *const usize,
        b: *const i32,
        b_strides: *const usize,
        c: *mut i32,
        c_strides: *const usize,
        shape: *const usize,
        ndim: i32,
        total_elements: usize,
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

    pub fn cpu_matmul_strided_f32(
        a: *const f32,
        a_stride_row: usize,
        a_stride_col: usize,
        b: *const f32,
        b_stride_row: usize,
        b_stride_col: usize,
        c: *mut f32,
        c_stride_row: usize,
        c_stride_col: usize,
        m: i32,
        n: i32,
        k: i32,
    );
    pub fn gpu_matmul_strided_f32(
        a: *const f32,
        a_stride_row: usize,
        a_stride_col: usize,
        b: *const f32,
        b_stride_row: usize,
        b_stride_col: usize,
        c: *mut f32,
        c_stride_row: usize,
        c_stride_col: usize,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
