// nnc — neural-net op surface.
// Public declarations for the SIMD/JIT-routed kernels (gelu, softmax,
// layernorm, dot, gemv, elementwise add) and the small graph-level
// fuser that collapses mul_mat -> bias-add [-> gelu] chains.

#pragma once

#include <cstddef>
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

// Fused per-head attention softmax + V matmul:
//   m = max_t scores[t]
//   w_t = exp(scores[t] - m)         (in place into `scores`)
//   S = sum_t w_t
//   out[i] = (sum_t w_t * V[t * v_stride + i]) / S        for i in [0, head_dim)
//
// Equivalent to:
//   nnc_softmax_f32_inplace(scores, n_t);
//   memset(out, 0, head_dim * sizeof(float));
//   for (t) for (i) out[i] += scores[t] * V[t*v_stride + i];
// but in a single pass over `scores`: avoids the second read of the
// softmax output (n_t * 4 bytes per head per layer per token) and the
// initial zero of `out`. `V` is laid out as `n_t` rows of `v_stride`
// floats with the head's V vector at offset 0 of each row.
void nnc_attn_softmax_v_f32(float* out, float* scores, const float* V,
                            size_t n_t, size_t v_stride, size_t head_dim);

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

// Apply (RMSNorm + per-channel gamma multiply) to `n_groups` contiguous
// length-`dim` vectors. Equivalent to:
//   for (g) { rmsnorm(y+g*dim, x+g*dim, dim); for (i) y[g*dim+i] *= gamma[i]; }
// but the gamma vector stays hot in L1 across all groups.
void nnc_rmsnorm_gamma_multi_f32(float* y, const float* x,
                                 size_t n_groups, size_t dim,
                                 const float* gamma, float eps);

// AVX2 / FMA dot product of two FP32 vectors. 4-accumulator unroll for
// 8*4 = 32-element strides; tail handled scalarly. Intended for the
// per-head attention Q.K dot (head_dim=256 / 512), where the JITed
// dot_f32 kernel's call overhead would dominate.
float nnc_dot_f32_simd(const float* a, const float* b, size_t n);

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

// Q8_0 split-layout gemv: y[r] = sum_b scales[r,b] * sum_{k in block b} qs[r,k]*x[k]
// for r in [0, rows). qs is row-major int8 of size rows*cols; scales is
// row-major fp32 of size rows*(cols/32). cols must be a positive multiple
// of 32. Routes to a JIT 1-row kernel (cols baked) and parallelises the
// row axis through the same worker pool as the BF16 path when rows is
// large enough to amortise dispatch.
void nnc_gemv_q8_0_f32x(const int8_t* qs, const float* scales,
                        const float* x, float* y,
                        uint32_t rows, uint32_t cols);

// In-place quantize a row-major BF16 weight matrix [rows x cols] into the
// Q8_0 split layout: writes `qs[rows*cols]` (int8) followed by
// `scales[rows*(cols/32)]` (fp32). cols must be a positive multiple of 32.
// Uses absmax-per-block scaling: scale = max(|w|)/127, q = round(w/scale).
void nnc_quantize_bf16_to_q8_0(const uint16_t* W_bf16, int8_t* qs,
                               float* scales, size_t rows, size_t cols);

// ---- K-quant dequantizers ---------------------------------------------
//
// All three layouts use a 256-element super-block with fp16 super-scales
// and 6-bit (Q4_K/Q5_K) or 8-bit (Q6_K) per-sub-block scales. Block
// sizes (matching ggml `block_q*_K`):
//   Q4_K = 144 bytes  ( 256 elems, 4-bit qs)
//   Q5_K = 176 bytes  ( 256 elems, 4-bit qs + 1-bit qh)
//   Q6_K = 210 bytes  ( 256 elems, 4-bit ql + 2-bit qh + i8 scales)
//
// `n_elements` must be a positive multiple of 256. `blocks` points at
// the first packed block; `dst` receives `n_elements` contiguous
// floats. Scalar reference implementations (no SIMD yet); used at load
// time to dequant K-quant weights into a denser format that the
// existing BF16 / Q8_0 gemv kernels can consume.
void nnc_dequantize_q4_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q5_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q6_k_to_f32(const void* blocks, float* dst, size_t n_elements);

// Dispatch by the GGUF / ggml type code (12=Q4_K, 13=Q5_K, 14=Q6_K).
// Returns false (and leaves dst untouched) for unsupported types.
bool nnc_dequantize_kquant_to_f32(uint32_t ggml_type, const void* blocks,
                                  float* dst, size_t n_elements);

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

// Fused gated-MLP activation:  y[i] = gelu(gate[i]) * up[i].
// Replaces a two-pass `nnc_gelu_f32(gate); nnc_mul_inplace_f32(gate, up)`
// pair, halving memory traffic on `gate` (one read + one write instead
// of two of each). Matches the Gemma 4 "FFN_GELU + PAR" inner step.
// `y` may alias `gate` or `up`.
void nnc_gelu_mul_f32(float* y, const float* gate, const float* up, size_t n);

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
                  uint32_t n_rot, float pos, float freq_base);

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
