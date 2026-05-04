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

// Emits  void gemv_bf16w_f32x(const bf16* W, const float* x, float* y);
// rcx=W, rdx=x, r8=y. rows and cols are baked into the emitted code as
// 32-bit immediates. cols must be a positive multiple of 8.
//
// Layout: y[r] = sum_{k=0..cols-1} bf16_to_fp32(W[r*cols + k]) * x[k].
//
// BF16 -> F32 is implemented as `vpmovzxwd` (8 u16 -> 8 u32) + `vpslld 16`
// (so the BF16 bit-pattern becomes the high half of the FP32 word). Same
// 4-accumulator unrolling as the FP16 builder when cols % 32 == 0.
void nnc_build_gemv_bf16w_f32x(jit_buffer& buf, uint32_t rows, uint32_t cols);

// Emits  void gemv_bf16w_f32x_4row(const bf16* W, const float* x, float* y);
// Same ABI as nnc_build_gemv_bf16w_f32x but processes 4 rows in parallel
// per outer iteration: the x[k..k+7] tile is loaded once per inner step
// and broadcast to 4 independent FMA accumulators (one per row), cutting
// x-side bandwidth by 4x. The 4 row partial-sums are reduced with a
// single 3-vhaddps tree and stored as a contiguous [a,b,c,d] xmm.
//
// Requires rows > 0 and rows % 4 == 0, cols > 0 and cols % 8 == 0.
void nnc_build_gemv_bf16w_f32x_4row(jit_buffer& buf, uint32_t rows, uint32_t cols);

// Emits a single-row Q8_0 dot-product kernel:
//   void gemv_q8_0_1row(const int8_t* qs, const float* x,
//                       float* y_out, const float* scales);
// Computes  *y_out = sum_b scales[b] * sum_{k in block b} qs[b*32+k] * x[b*32+k]
// and writes it to *y_out. cols must be a positive multiple of 32. The
// caller (worker pool) iterates rows externally, advancing qs by `cols`
// bytes, scales by `(cols/32)*4` bytes, and y by 4 bytes per row. Inner
// loop processes one Q8_0 block (32 cols) per iteration with 4 unrolled
// 8-col FMA steps.
//
// Win64 ABI: rcx=qs, rdx=x, r8=y_out, r9=scales. Saves rsi+rdi.
void nnc_build_gemv_q8_0_f32x_1row(jit_buffer& buf, uint32_t cols);
