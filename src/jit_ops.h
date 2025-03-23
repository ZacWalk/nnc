// nnc — high-level kernel builders.

#pragma once

#include <cstdint>

class jit_buffer;

// Emits  float dot_f32(const float* a, const float* b, size_t n);
// rcx=a, rdx=b, r8=n, return in xmm0. n must be a multiple of 8.
void nnc_build_dot_f32(jit_buffer& buf);

// Emits  void gemv_f32(const float* W, const float* x, float* y);
// rcx=W, rdx=x, r8=y. rows and cols are baked into the emitted code as
// 32-bit immediates. cols must be a positive multiple of 8.
//
// Layout: y[r] = sum_{k=0..cols-1} W[r*cols + k] * x[k].
//
// Saves rsi+rdi in prologue (Win64 nonvolatile) so the kernel can keep
// all data pointers in low-8 GPRs.
void nnc_build_gemv_f32(jit_buffer& buf, uint32_t rows, uint32_t cols);

// Emits  float dot_f16_to_f32(const fp16* x, const fp16* y);
// rcx=x, rdx=y, return in xmm0. n is baked. Requires n > 0 and n % 32 == 0.
// Fully unrolled. Uses F16C vcvtph2ps to expand 8 halves -> 8 floats and
// FMA into 4 ymm accumulators (sum0..sum3) for better ILP.
void nnc_build_dot_f16_to_f32(jit_buffer& buf, uint32_t n);

// Emits  void gemv_f16w_f32x(const fp16* W, const float* x, float* y);
// rcx=W, rdx=x, r8=y. rows and cols are baked into the emitted code as
// 32-bit immediates. cols must be a positive multiple of 8.
//
// Layout: y[r] = sum_{k=0..cols-1} fp16_to_fp32(W[r*cols + k]) * x[k].
//
// per-row vec_dot_f16 and skips its FP32->FP16 packing of x
// entirely. Saves rsi+rdi in prologue (Win64 nonvolatile).
void nnc_build_gemv_f16w_f32x(jit_buffer& buf, uint32_t rows, uint32_t cols);
