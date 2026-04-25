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

// RMSNorm over n contiguous floats. No mean subtraction:
//   r = sqrt(mean(x^2) + eps);  y[i] = x[i] / r.
// y and x may alias. Used by Gemma / Llama-style models. The (optional)
// per-channel learned scale is applied by the caller.
void nnc_rmsnorm_f32(float* y, const float* x, size_t n, float eps);

// Fused FP16-weights, FP32-activations gemv:
//   y[r] = sum_{k=0..cols-1} fp16_to_fp32(W[r*cols + k]) * x[k]   for r in [0, rows).
// W is FP16 (uint16_t / nnc_fp16_t), x and y are FP32. Routes to a JITted
// AVX2+F16C+FMA kernel cached by (rows, cols) when cols is a multiple of 8.
void nnc_gemv_f16w_f32x(const void* W, const float* x, float* y,
                        uint32_t rows, uint32_t cols);

// Same shape contract as nnc_gemv_f16w_f32x but W is BF16 (uint16_t with
// the upper 16 bits of an IEEE-754 binary32). AVX2 path uses
// vpmovzxwd + vpslld to inflate to F32 then vfmadd231ps. No JIT yet —
// the inner loop is hand-vectorised.
void nnc_gemv_bf16w_f32x(const void* W, const float* x, float* y,
                         uint32_t rows, uint32_t cols);

// Streaming "BF16-weight gemv + argmax" — computes y[r] = sum_k bf16->f32(W[r,k])*x[k]
// for all r in [0, rows) and returns argmax_r y[r] without ever materialising
// the full y[] vector. Used for the lm_head / final logits projection when the
// caller only needs the greedy-decode token id (no top-k / top-p sampling).
//
// Walks the row axis in groups of 4 (re-using the cached BF16 4-row JIT kernel),
// updating a running (best_val, best_idx) pair from each 4-element batch. This
// avoids a full vocab-sized write (~1 MB at vocab=262144) and the subsequent
// softcap + linear-scan argmax passes. Softcap is monotonic so it has no effect
// on argmax and is omitted entirely.
//
// rows must be > 0 and a multiple of 4; cols must be > 0 and a multiple of 8.
// Falls back to nnc_gemv_bf16w_f32x + scratch + scalar argmax when those
// preconditions don't hold.
int nnc_gemv_bf16w_argmax_f32x(const void* W, const float* x,
                               uint32_t rows, uint32_t cols);

// Fused SwiGLU activation used by Gemma / Llama MLPs:
//   y[i] = silu(gate[i]) * up[i]   where silu(x) = x * sigmoid(x).
// y may alias gate or up.
void nnc_swiglu_f32(float* y, const float* gate, const float* up, size_t n);

// SIMD elementwise add: y[i] += b[i] for i in [0, n).
void nnc_add_inplace_f32(float* y, const float* b, size_t n);

// SIMD elementwise multiply: y[i] *= s[i] for i in [0, n). Used to apply
// a learned per-channel scale (e.g. Gemma's RMSNorm gamma) right after
// nnc_rmsnorm_f32.
void nnc_mul_inplace_f32(float* y, const float* s, size_t n);

// Rotary position embedding (NeoX/half-pair convention used by Gemma /
// Llama / Mistral). For one token at position `pos`, applies an in-place
// rotation to the first `n_rot` lanes of each head's `head_dim`-vector:
//
//   theta_i = pos * freq_base^(-2i / n_rot)        for i in [0, n_rot/2)
//   x'[i]          =  cos(theta_i) * x[i]          - sin(theta_i) * x[i + n_rot/2]
//   x'[i+n_rot/2]  =  sin(theta_i) * x[i]          + cos(theta_i) * x[i + n_rot/2]
//
// Lanes [n_rot, head_dim) are passed through unchanged. `n_heads` heads
// are processed (each `head_dim` floats apart). `freq_base` is the RoPE
// theta base (Gemma uses 1e6 for global layers, 1e4 for sliding-window
// layers).
void nnc_rope_f32(float* x, uint32_t n_heads, uint32_t head_dim,
                  uint32_t n_rot, int32_t pos, float freq_base);

// Logit soft-cap (Gemma final-layer & some attention-layer outputs):
//   y[i] = tanh(x[i] / cap) * cap.
// y and x may alias.
void nnc_softcap_f32(float* y, const float* x, size_t n, float cap);

// Look up one row of a BF16 embedding table and convert it to FP32:
//   y[i] = bf16_to_f32(table[token_id * n_embd + i]) * scale
// for i in [0, n_embd). `scale` is typically sqrt(n_embd) for Gemma.
void nnc_embed_row_bf16(float* y, const void* table, int token_id,
                        size_t n_embd, float scale);

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
