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
// but the accumulation writes `out` on the first V row instead of
// zeroing it first, and keeps the weighted sum in one sweep. `V` is
// laid out as `n_t` rows of `v_stride` floats with the head's V vector
// at offset 0 of each row.
void nnc_attn_softmax_v_f32(float* out, float* scores, const float* V,
                            size_t n_t, size_t v_stride, size_t head_dim);

// Same, for `n_heads` query heads that all attend to the SAME K/V rows
// (grouped-query attention with n_head > n_head_kv, which is the norm:
// Gemma 3 1B is 4 query heads to 1 KV head).
//
// `scores` holds n_heads rows of n_t floats at `scores_stride` apart;
// `out` holds n_heads vectors of head_dim floats at `out_stride` apart.
//
// The point is the loop order: `t` runs outermost so each V row is pulled
// from memory once and then reused across all heads from L1. Calling the
// single-head version in a loop instead re-streams the whole V cache once
// per head, which at long context is pure wasted DRAM bandwidth.
// Numerically identical to per-head calls — each head still accumulates
// in increasing t order.
void nnc_attn_softmax_v_multi_f32(float* out, size_t out_stride,
                                  float* scores, size_t scores_stride,
                                  const float* V, size_t n_t, size_t v_stride,
                                  size_t head_dim, size_t n_heads);

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

// BF16-weights, FP32-activations gemv:
//   y[r] = sum_{k=0..cols-1} bf16_to_fp32(W[r*cols + k]) * x[k]   for r in [0, rows).
// W is BF16 (uint16_t holding the upper 16 bits of an IEEE-754 binary32),
// x and y are FP32. Routes to a JITted AVX2+FMA kernel cached by
// (rows, cols) when cols is a multiple of 8 (vpmovzxwd + vpslld 16 +
// vfmadd231ps); scalar fallback otherwise.
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
// row-major BF16 of size rows*(cols/32). cols must be a positive multiple
// of 32. Routes to a JIT 1-row kernel (cols baked) and parallelises the
// row axis through the same worker pool as the BF16 path when rows is
// large enough to amortise dispatch.
void nnc_gemv_q8_0_f32x(const int8_t* qs, const uint16_t* scales,
                        const float* x, float* y,
                        uint32_t rows, uint32_t cols);

// Q4 (nnc split 4-bit) gemv. Reconstruction is w = scale*q + bias with q
// an unsigned nibble in [0, 15], so
//   y[r] = sum_b [ scales[r,b] * sum_{k in b} q[r,k]*x[k] + biases[r,b] * S[b] ]
// where S[b] = sum_{k in b} x[k] depends only on x. `xsum` must hold those
// cols/32 block sums (see nnc_block_sums_f32); passing it in keeps the
// bias term out of the JIT kernel and off the per-row critical path.
//
// qs is row-major packed nibbles of size rows*cols/2; scales and biases
// are row-major BF16 of size rows*(cols/32). BF16 rather than FP32 halves
// the metadata traffic (0.75 -> 0.625 bytes/weight), and rather than FP16
// because a block scale can be arbitrarily small — BF16 keeps FP32's
// exponent range so it cannot flush to subnormal. cols must be a positive
// multiple of 32.
void nnc_gemv_q4_s_f32x(const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, const float* x,
                        const float* xsum, float* y,
                        uint32_t rows, uint32_t cols);

// out[b] = sum of x[b*32 .. b*32+31], for b in [0, n/32). n must be a
// positive multiple of 32.
void nnc_block_sums_f32(const float* x, float* out, size_t n);

// ---- batched gemv (prefill) -------------------------------------------
//
// Same maths as the single-vector gemvs above, but applied to `n_batch`
// activation vectors against one pass over the weights. The loop order is
// row-major outer / batch inner, so each weight row is pulled from DRAM
// once and then reused from L1 for every vector in the batch — which is
// the entire point, since decode and prefill are both bandwidth bound.
//
// `X` holds n_batch vectors of `cols` floats at `x_stride` floats apart;
// `Y` holds n_batch vectors of `rows` floats at `y_stride` floats apart.
// n_batch == 1 is exactly the unbatched call.
void nnc_gemv_bf16w_f32x_batch(const void* W, const float* X, size_t x_stride,
                               float* Y, size_t y_stride,
                               uint32_t rows, uint32_t cols, uint32_t n_batch);

void nnc_gemv_q8_0_f32x_batch(const int8_t* qs, const uint16_t* scales,
                              const float* X, size_t x_stride,
                              float* Y, size_t y_stride,
                              uint32_t rows, uint32_t cols, uint32_t n_batch);

// `XSUM` holds the per-32-block sums of each X vector (see
// nnc_block_sums_f32), n_batch runs of cols/32 floats at xsum_stride apart.
void nnc_gemv_q4_s_f32x_batch(const uint8_t* qs, const uint16_t* scales,
                              const uint16_t* biases,
                              const float* X, size_t x_stride,
                              const float* XSUM, size_t xsum_stride,
                              float* Y, size_t y_stride,
                              uint32_t rows, uint32_t cols, uint32_t n_batch);

// In-place quantize a row-major BF16 weight matrix [rows x cols] into the
// Q8_0 split layout: writes `qs[rows*cols]` (int8) followed by
// `scales[rows*(cols/32)]` (BF16). cols must be a positive multiple of 32.
// Uses absmax-per-block scaling: scale = max(|w|)/127, q = round(w/scale).
void nnc_quantize_bf16_to_q8_0(const uint16_t* W_bf16, int8_t* qs,
                               uint16_t* scales, size_t rows, size_t cols);

// Same, from FP32 input, over a flat run of `n` elements (n must be a
// positive multiple of 32). Because Q8_0 blocks are 32 wide and every
// quantized weight matrix has `cols % 32 == 0`, a flat walk produces
// exactly the row-major block order the split layout expects — so this
// can be called incrementally on chunks of a larger matrix.
void nnc_quantize_f32_to_q8_0(const float* src, int8_t* qs,
                              uint16_t* scales, size_t n);

// Quantize a flat run of `n` FP32 weights (n a positive multiple of 32)
// into the nnc split 4-bit layout: writes `qs[n/2]` packed nibbles plus
// one BF16 `scale` and one BF16 `bias` per 32-element block, such that
//   w[k] ~= scales[k/32] * q[k] + biases[k/32],   q in [0, 15].
// Asymmetric (min/max) rather than absmax, which is what makes 4 bits
// usable: scale = (max-min)/15, bias = min.
//
// Nibble packing within a block: byte i holds element i in its low
// nibble and element i+16 in its high nibble.
void nnc_quantize_f32_to_q4_s(const float* src, uint8_t* qs,
                              uint16_t* scales, uint16_t* biases, size_t n);

// Returns sum_i bf16_to_f32(a[i]) * b[i]. Used for the Q4 bias term.
float nnc_dot_bf16_f32_simd(const uint16_t* a, const float* b, size_t n);

// ---- load-time dequantizers -------------------------------------------
//
// Q8_0 uses a 32-element block: { fp16 d; int8 qs[32] } = 34 bytes.
// Q4_K/Q5_K/Q6_K use a 256-element super-block with fp16 super-scales
// and 6-bit (Q4_K/Q5_K) or 8-bit (Q6_K) per-sub-block scales. Block
// sizes (matching ggml `block_q*`):
//   Q8_0 =  34 bytes  (  32 elems)
//   Q4_K = 144 bytes  ( 256 elems, 4-bit qs)
//   Q5_K = 176 bytes  ( 256 elems, 4-bit qs + 1-bit qh)
//   Q6_K = 210 bytes  ( 256 elems, 4-bit ql + 2-bit qh + i8 scales)
//
// `n_elements` must be a positive multiple of the type's block size.
// `blocks` points at the first packed block; `dst` receives
// `n_elements` contiguous floats. Scalar implementations: these run
// once at load time, feeding the BF16 / Q8_0 gemv kernels.
void nnc_dequantize_q8_0_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q4_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q5_k_to_f32(const void* blocks, float* dst, size_t n_elements);
void nnc_dequantize_q6_k_to_f32(const void* blocks, float* dst, size_t n_elements);

// Elements per storage block for a GGUF / ggml quantized type code.
// Returns 0 for types this build cannot decode.
uint32_t nnc_quant_block_elems(uint32_t ggml_type);

// Dispatch by the GGUF / ggml type code (8=Q8_0, 12=Q4_K, 13=Q5_K,
// 14=Q6_K). Returns false (and leaves dst untouched) for other types.
bool nnc_dequantize_to_f32(uint32_t ggml_type, const void* blocks,
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

// Same, from a Q8_0 split-layout table: qs is row-major int8 of size
// rows*n_embd, scales is row-major BF16 of size rows*(n_embd/32).
// n_embd must be a positive multiple of 32.
void nnc_embed_row_q8_0(float* y, const int8_t* qs, const uint16_t* scales,
                        int token_id, size_t n_embd, float scale);

// Same, from a Q4 split-layout table (BF16 scales/biases).
void nnc_embed_row_q4_s(float* y, const uint8_t* qs, const uint16_t* scales,
                        const uint16_t* biases, int token_id,
                        size_t n_embd, float scale);
