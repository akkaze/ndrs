#pragma once
#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    // 加法 (float32)
    EXPORT void cpu_add_f32(const float* a, const float* b, float* c, size_t n);
    EXPORT int gpu_add_f32(const float* a, const float* b, float* c, size_t n, void* stream);

    // 加法 (int32)
    EXPORT void cpu_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n);
    EXPORT int gpu_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n, void* stream);

    // 跨步拷贝 (通用，字节粒度)
    EXPORT void cpu_strided_copy(const uint8_t* src, size_t src_offset,
                                 const size_t* src_strides, int ndim,
                                 const size_t* shape,
                                 uint8_t* dst, size_t dst_offset,
                                 const size_t* dst_strides,
                                 size_t elem_size, size_t total_elements);

    EXPORT int gpu_strided_copy(const uint8_t* src, size_t src_offset,
                                const size_t* src_strides, int ndim,
                                const size_t* shape,
                                uint8_t* dst, size_t dst_offset,
                                const size_t* dst_strides,
                                size_t elem_size, size_t total_elements,
                                void* stream);

    // 连续化
    EXPORT void cpu_contiguous(const uint8_t* src, size_t src_offset,
                               const size_t* src_strides, int ndim,
                               const size_t* shape,
                               uint8_t* dst, size_t elem_size,
                               size_t total_elements);

    EXPORT int gpu_contiguous(const uint8_t* src, size_t src_offset,
                              const size_t* src_strides, int ndim,
                              const size_t* shape,
                              uint8_t* dst, size_t elem_size,
                              size_t total_elements,
                              void* stream);
}