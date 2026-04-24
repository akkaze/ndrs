#include "kernel_interface.h"
#include <omp.h>
#include <cstring>
#include <cstdint>

#ifdef _MSC_VER
    #define OMP_COLLAPSE(x)
    #define OMP_FOR_SIGNED
#else
    #define OMP_COLLAPSE(x) collapse(x)
#endif

template<typename T>
struct add_op { T operator()(T x, T y) const { return x + y; } };
template<typename T>
struct sub_op { T operator()(T x, T y) const { return x - y; } };
template<typename T>
struct mul_op { T operator()(T x, T y) const { return x * y; } };
template<typename T>
struct div_op { T operator()(T x, T y) const { return x / y; } };

template<typename T, typename Op>
void cpu_strided_binary(const T* a, const size_t* a_strides,
                        const T* b, const size_t* b_strides,
                        T* c, const size_t* c_strides,
                        const size_t* shape, int ndim,
                        size_t total_elements, Op op) {
    #pragma omp parallel for
    for (int64_t idx = 0; idx < total_elements; ++idx) {
        size_t a_off = 0, b_off = 0, c_off = 0;
        size_t temp = idx;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t i = temp % shape[d];
            temp /= shape[d];
            a_off += i * a_strides[d];
            b_off += i * b_strides[d];
            c_off += i * c_strides[d];
        }
        // 字节偏移转元素索引
        c[c_off / sizeof(T)] = op(a[a_off / sizeof(T)], b[b_off / sizeof(T)]);
    }
}

#define DEFINE_CPU_STRIDED_BINARY(T, op_name, op_type, type_name) \
    void cpu_strided_##op_name##_##type_name( \
        const T* a, const size_t* a_strides, \
        const T* b, const size_t* b_strides, \
        T* c, const size_t* c_strides, \
        const size_t* shape, int ndim, \
        size_t total_elements) \
    { \
        cpu_strided_binary<T, op_type<T>>(a, a_strides, b, b_strides, c, c_strides, shape, ndim, total_elements, op_type<T>()); \
    }

DEFINE_CPU_STRIDED_BINARY(float, add, add_op, f32)
DEFINE_CPU_STRIDED_BINARY(float, sub, sub_op, f32)
DEFINE_CPU_STRIDED_BINARY(float, mul, mul_op, f32)
DEFINE_CPU_STRIDED_BINARY(float, div, div_op, f32)

// int32_t 类型
DEFINE_CPU_STRIDED_BINARY(int32_t, add, add_op, i32)
DEFINE_CPU_STRIDED_BINARY(int32_t, sub, sub_op, i32)
DEFINE_CPU_STRIDED_BINARY(int32_t, mul, mul_op, i32)
DEFINE_CPU_STRIDED_BINARY(int32_t, div, div_op, i32)


void cpu_strided_copy(const uint8_t* src, size_t src_offset,
                      const size_t* src_strides, int ndim,
                      const size_t* shape,
                      uint8_t* dst, size_t dst_offset,
                      const size_t* dst_strides,
                      size_t elem_size, size_t total_elements) {
    #pragma omp parallel for
    for (int64_t idx = 0; idx < total_elements; ++idx) {
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
    for (int64_t idx = 0; idx < total_elements; ++idx) {
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

void cpu_matmul_strided_f32(
    const float* A, size_t a_stride_row, size_t a_stride_col,
    const float* B, size_t b_stride_row, size_t b_stride_col,
    float* C, size_t c_stride_row, size_t c_stride_col,
    int M, int N, int K) 
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float* a_ptr = (const float*)((const char*)A + i * a_stride_row + k * a_stride_col);
                const float* b_ptr = (const float*)((const char*)B + k * b_stride_row + j * b_stride_col);
                sum += *a_ptr * *b_ptr;
            }
            float* c_ptr = (float*)((char*)C + i * c_stride_row + j * c_stride_col);
            *c_ptr = sum;
        }
    }
}