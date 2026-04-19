#include "kernel_interface.h"
#include <omp.h>
#include <cstring>

void cpu_add_f32(const float* a, const float* b, float* c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void cpu_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void cpu_strided_copy(const uint8_t* src, size_t src_offset,
                      const size_t* src_strides, int ndim,
                      const size_t* shape,
                      uint8_t* dst, size_t dst_offset,
                      const size_t* dst_strides,
                      size_t elem_size, size_t total_elements) {
    #pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) {
        size_t src_off = src_offset;
        size_t dst_off = dst_offset;
        size_t temp = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t i = temp % shape[d];
            temp /= shape[d];
            src_off += i * src_strides[d];
            dst_off += i * dst_strides[d];
        }
        std::memcpy(dst + dst_off, src + src_off, elem_size);
    }
}

void cpu_contiguous(const uint8_t* src, size_t src_offset,
                    const size_t* src_strides, int ndim,
                    const size_t* shape,
                    uint8_t* dst, size_t elem_size,
                    size_t total_elements) {
    #pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) {
        size_t src_off = src_offset;
        size_t temp = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t i = temp % shape[d];
            temp /= shape[d];
            src_off += i * src_strides[d];
        }
        std::memcpy(dst + idx * elem_size, src + src_off, elem_size);
    }
}