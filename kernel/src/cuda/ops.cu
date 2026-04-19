#include "kernel_interface.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__global__ void add_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void add_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

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

int gpu_add_f32(const float* a, const float* b, float* c, size_t n, void* stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_f32_kernel<<<blocks, threads, 0, s>>>(a, b, c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_add_f32 kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
}

int gpu_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n, void* stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_i32_kernel<<<blocks, threads, 0, s>>>(a, b, c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_add_i32 kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
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