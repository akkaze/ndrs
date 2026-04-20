#include "kernel_interface.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// 定义运算的仿函数（也可用 lambda，但模板参数需要类型）
template<typename T>
struct add_op { __device__ T operator()(T x, T y) const { return x + y; } };

template<typename T>
struct sub_op { __device__ T operator()(T x, T y) const { return x - y; } };

template<typename T>
struct mul_op { __device__ T operator()(T x, T y) const { return x * y; } };

template<typename T>
struct div_op { __device__ T operator()(T x, T y) const { return x / y; } };

// 通用带步长的 kernel
template<typename T, typename Op>
__global__ void strided_binary_kernel(
    const T* a, const size_t* a_strides,
    const T* b, const size_t* b_strides,
    T* c, const size_t* c_strides,
    const size_t* shape, int ndim,
    size_t total_elements, Op op) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    size_t a_off = 0, b_off = 0, c_off = 0;
    size_t temp = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        size_t i = temp % shape[d];
        temp /= shape[d];
        a_off += i * a_strides[d];
        b_off += i * b_strides[d];
        c_off += i * c_strides[d];
    }
    // 将字节偏移转换为元素索引
    c[c_off / sizeof(T)] = op(a[a_off / sizeof(T)], b[b_off / sizeof(T)]);
}


#define DEFINE_STRIDED_BINARY_KERNEL(T, op_name, op_type, type_name) \
    int gpu_strided_##op_name##_##type_name( \
        const T* a, const size_t* a_strides, \
        const T* b, const size_t* b_strides, \
        T* c, const size_t* c_strides, \
        const size_t* shape, int ndim, \
        size_t total_elements, void* stream) \
    { \
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream); \
        int threads = 256; \
        int blocks = (total_elements + threads - 1) / threads; \
        op_type<T> op; \
        strided_binary_kernel<T, op_type<T>><<<blocks, threads, 0, s>>>( \
            a, a_strides, b, b_strides, c, c_strides, shape, ndim, total_elements, op); \
        return cudaGetLastError(); \
    }


// 实例化需要的类型和运算
DEFINE_STRIDED_BINARY_KERNEL(float, add, add_op, f32)
DEFINE_STRIDED_BINARY_KERNEL(float, sub, sub_op, f32)
DEFINE_STRIDED_BINARY_KERNEL(float, mul, mul_op, f32)
DEFINE_STRIDED_BINARY_KERNEL(float, div, div_op, f32)
DEFINE_STRIDED_BINARY_KERNEL(int32_t, add, add_op, i32)


__global__ void strided_copy_kernel(const uint8_t* src, size_t src_offset,
                                    const size_t* src_strides, int ndim,
                                    const size_t* shape,
                                    uint8_t* dst, size_t dst_offset,
                                    const size_t* dst_strides,
                                    size_t elem_size, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    size_t src_off = src_offset;
    size_t dst_off = dst_offset;
    size_t temp = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        size_t i = temp % shape[d];
        temp /= shape[d];
        src_off += i * src_strides[d];
        dst_off += i * dst_strides[d];
    }
    for (size_t b = 0; b < elem_size; ++b) {
        dst[dst_off + b] = src[src_off + b];
    }
}

__global__ void contiguous_kernel(const uint8_t* src, size_t src_offset,
                                  const size_t* src_strides, int ndim,
                                  const size_t* shape,
                                  uint8_t* dst, size_t elem_size,
                                  size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    size_t src_off = src_offset;
    size_t temp = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        size_t i = temp % shape[d];
        temp /= shape[d];
        src_off += i * src_strides[d];
    }
    for (size_t b = 0; b < elem_size; ++b) {
        dst[idx * elem_size + b] = src[src_off + b];
    }
}

int gpu_strided_copy(const uint8_t* src, size_t src_offset,
                     const size_t* src_strides, int ndim,
                     const size_t* shape,
                     uint8_t* dst, size_t dst_offset,
                     const size_t* dst_strides,
                     size_t elem_size, size_t total_elements,
                     void* stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    strided_copy_kernel<<<blocks, threads, 0, s>>>(
        src, src_offset, src_strides, ndim, shape,
        dst, dst_offset, dst_strides,
        elem_size, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_strided_copy kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
}

int gpu_contiguous(const uint8_t* src, size_t src_offset,
                   const size_t* src_strides, int ndim,
                   const size_t* shape,
                   uint8_t* dst, size_t elem_size,
                   size_t total_elements,
                   void* stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    contiguous_kernel<<<blocks, threads, 0, s>>>(
        src, src_offset, src_strides, ndim, shape,
        dst, elem_size, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_contiguous kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
}

__global__ void matmul_strided_f32_kernel(
    const float* A, size_t a_stride_row, size_t a_stride_col,
    const float* B, size_t b_stride_row, size_t b_stride_col,
    float* C, size_t c_stride_row, size_t c_stride_col,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float* a_ptr = (const float*)((const char*)A + row * a_stride_row + k * a_stride_col);
        const float* b_ptr = (const float*)((const char*)B + k * b_stride_row + col * b_stride_col);
        sum += *a_ptr * *b_ptr;
    }
    float* c_ptr = (float*)((char*)C + row * c_stride_row + col * c_stride_col);
    *c_ptr = sum;
}

int gpu_matmul_strided_f32(
    const float* A, size_t a_stride_row, size_t a_stride_col,
    const float* B, size_t b_stride_row, size_t b_stride_col,
    float* C, size_t c_stride_row, size_t c_stride_col,
    int M, int N, int K, void* stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_strided_f32_kernel<<<blocks, threads, 0, s>>>(
        A, a_stride_row, a_stride_col,
        B, b_stride_row, b_stride_col,
        C, c_stride_row, c_stride_col,
        M, N, K);
    return cudaGetLastError();
}