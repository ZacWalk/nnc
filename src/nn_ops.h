// nnc — neural-net op surface.
// Public declarations for the SIMD/JIT-routed kernels (gelu, softmax,
// layernorm, dot, gemv, elementwise add) and the small graph-level
// fuser that collapses mul_mat -> bias-add [-> gelu] chains.

#pragma once

#include <cstdint>

// y[i] = 0.5 * x[i] * (1 + tanh( sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3) ))
// FP32 in / FP32 out. y and x may alias (out-of-place use is also fine).
void nnc_gelu_f32(float* y, const float* x, size_t n);

// Returns sum_i  fp16_to_fp32(x[i]) * fp16_to_fp32(y[i])  for i in [0, n).
// x and y are arrays of IEEE 754 binary16 (nnc_fp16_t / uint16_t) values.
// Routes through a JITted F16C+FMA kernel when n > 0 and n % 32 == 0; falls
// back to a scalar reference for any other size. Thread-safety: the kernel
// cache is guarded by an internal mutex on the build path; cached pointers
// are read lock-free.
float nnc_dot_f16_to_f32(const void* x, const void* y, size_t n);

// In-place numerically-stable softmax over n contiguous floats:
//   m = max(p);  p[i] = exp(p[i] - m) / sum_j exp(p[j] - m).
// -INFINITY entries map to 0 (used for causal attention masks).
void nnc_softmax_f32_inplace(float* p, size_t n);

// LayerNorm over n contiguous floats (mean 0, variance 1, then optional
// affine is applied by a separate op
//   m  = mean(x); v = mean((x-m)^2);  y[i] = (x[i] - m) / sqrt(v + eps).
// y and x may alias.
void nnc_layernorm_f32(float* y, const float* x, size_t n, float eps);

// Fused FP16-weights, FP32-activations gemv:
//   y[r] = sum_{k=0..cols-1} fp16_to_fp32(W[r*cols + k]) * x[k]   for r in [0, rows).
// W is FP16 (uint16_t / nnc_fp16_t), x and y are FP32. Routes to a JITted
// AVX2+F16C+FMA kernel cached by (rows, cols) when cols is a multiple of 8.
void nnc_gemv_f16w_f32x(const void* W, const float* x, float* y,
                        uint32_t rows, uint32_t cols);

// SIMD elementwise add: y[i] += b[i] for i in [0, n).
void nnc_add_inplace_f32(float* y, const float* b, size_t n);

// --- graph-level fusion (mul_mat -> repeat(bias) -> add) ---
struct nnc_cgraph;
struct nnc_tensor;

void nnc_graph_prefuse(const struct nnc_cgraph* g);
bool nnc_should_skip(const struct nnc_tensor* node);
const float* nnc_fused_bias_for(const struct nnc_tensor* mul_mat_node);
// When non-null, the fused mul_mat must write its output here (the ADD's
// destination buffer) so downstream nodes consuming the ADD see the result.
void* nnc_fused_dst_for(const struct nnc_tensor* mul_mat_node);
// True when this mul_mat has a fused trailing GELU (apply after bias add).
bool nnc_fused_gelu_for(const struct nnc_tensor* mul_mat_node);
