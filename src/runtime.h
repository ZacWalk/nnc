// nnc runtime: arena allocator, tensor descriptors, and the dtype table
// used by the Gemma GGUF loader. Computation lives in nn_ops.cpp (SIMD)
// and jit_ops.cpp (JIT); gemma.cpp drives those kernels directly.

#pragma once

#include <cstdint>
#include <cstring>

#define NNC_MAX_DIMS 4

#ifndef NNC_ASSERT
// Custom assert: print "ASSERT failed: <expr> at <file>:<line>" to stderr,
// flush, and abort the process. No modal message box.
[[noreturn]] void nnc_assert_fail(const char* expr, const char* file, int line);
#define NNC_ASSERT(x) ((x) ? (void)0 : nnc_assert_fail(#x, __FILE__, __LINE__))
#endif

using nnc_bf16_t = uint16_t;

enum nnc_type
{
	NNC_TYPE_F32 = 0,
	NNC_TYPE_F16 = 1,
	NNC_TYPE_I32 = 2,
	NNC_TYPE_BF16 = 3,
	// Q8_0 (split layout): row r of an [rows x cols] weight matrix is
	// stored as `int8 qs[rows*cols]` followed by `float scales[rows*cols/32]`
	// in the same allocation. `tensor->data` points at `qs`; the kernel
	// computes the scales pointer from rows*cols. Block size is 32; the
	// quantizer fills cols-multiple-of-32 only.
	NNC_TYPE_Q8_0 = 4,
	// Q4 (split layout): `uint8 qs[rows*cols/2]` packed nibbles, then
	// `bf16 scales[rows*cols/32]`, then `bf16 biases[rows*cols/32]`, all
	// in one allocation. Reconstruction is w = scale*q + bias with q an
	// unsigned nibble in [0, 15]. Block size is 32; byte i of a block holds
	// element i in its low nibble and element i+16 in its high nibble.
	NNC_TYPE_Q4_S = 5,
	NNC_TYPE_COUNT,
};

// --- bf16 helpers (IEEE 754 truncation: u32(f32) >> 16) ---
static inline float nnc_bf16_to_f32(const nnc_bf16_t v)
{
	const uint32_t u = static_cast<uint32_t>(v) << 16;
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

static inline nnc_bf16_t nnc_f32_to_bf16(const float f)
{
	uint32_t u;
	std::memcpy(&u, &f, 4);
	// round-to-nearest-even
	const uint32_t rounding_bias = 0x7fff + ((u >> 16) & 1);
	return static_cast<nnc_bf16_t>((u + rounding_bias) >> 16);
}

// AVX2 batched bf16 -> f32 conversion (tail handled scalarly).
void nnc_bf16_to_f32_row(const nnc_bf16_t* src, float* dst, size_t n);

// A weight or activation view. `data` usually points into the GGUF mmap
// (or into a loader-owned dequant / Q8_0 buffer); the descriptor itself
// lives in an nnc_context arena.
struct nnc_tensor
{
	nnc_type type;
	int n_dims;
	int ne[NNC_MAX_DIMS]; // shape (ne[0] = fastest-varying)
	size_t nb[NNC_MAX_DIMS]; // strides in bytes
	void* data;
};

struct nnc_init_params
{
	size_t mem_size;
	void* mem_buffer; // null => malloc internally
};

struct nnc_context;

// --- arena lifecycle ---
struct nnc_context* nnc_init(struct nnc_init_params params);
void nnc_free(struct nnc_context* ctx);

// --- type metadata ---
size_t nnc_type_size(nnc_type t);
size_t nnc_nbytes(const struct nnc_tensor* t);
int64_t nnc_nelements(const struct nnc_tensor* t);

// --- tensor descriptor allocation (into the arena) ---
struct nnc_tensor* nnc_new_tensor_1d(struct nnc_context* ctx, nnc_type t, int ne0);

// --- timing helpers (microseconds since process start) ---
void nnc_time_init();
int64_t nnc_time_us();
int64_t nnc_time_ns();

// --- phase profiler -----------------------------------------------------
//
// Fixed-slot, allocation-free wall-clock accumulator. Off by default; the
// `--perf` CLI flag turns it on and prints a breakdown after generation.
// Scopes must not overlap within a slot; nesting different slots is fine
// but the outer slot's time then includes the inner one (the report marks
// which slots are aggregates).
enum nnc_perf_slot
{
	NNC_PERF_EMBED = 0, // token embedding row lookup
	NNC_PERF_PLE_PREP, // per-layer-input embedding table + projection
	NNC_PERF_NORM, // RMSNorm + gamma (all sites)
	NNC_PERF_ATTN_QKV, // Q/K/V projections
	NNC_PERF_ATTN_ROPE, // per-head Q/K norms + RoPE
	NNC_PERF_ATTN_CORE, // Q.K scores + fused softmax*V
	NNC_PERF_ATTN_OUT, // attention output projection
	NNC_PERF_FFN_GATE_UP, // gate + up projections
	NNC_PERF_FFN_ACT, // gelu(gate)*up / swiglu
	NNC_PERF_FFN_DOWN, // down projection
	NNC_PERF_PLE_LAYER, // per-layer inp_gate + proj
	NNC_PERF_LM_HEAD, // final logits projection (+ argmax)
	NNC_PERF_COUNT,
};

extern bool g_nnc_perf_on;

inline bool nnc_perf_enabled() { return g_nnc_perf_on; }
void nnc_perf_enable(bool on);
void nnc_perf_reset();
void nnc_perf_add(nnc_perf_slot slot, int64_t ns, uint64_t bytes);
// Prints a per-slot table (ms, % of measured, calls, GB/s) plus totals.
// `n_tokens` scales the per-token columns; pass 0 to omit them.
void nnc_perf_report(const char* title, int n_tokens);

// RAII scope. `bytes` is the weight/activation traffic attributed to the
// scope, used to derive an effective bandwidth in the report.
struct nnc_perf_scope
{
	explicit nnc_perf_scope(const nnc_perf_slot s, const uint64_t bytes = 0)
		: slot_(s), bytes_(bytes), t0_(g_nnc_perf_on ? nnc_time_ns() : 0)
	{
	}

	~nnc_perf_scope()
	{
		if (g_nnc_perf_on) nnc_perf_add(slot_, nnc_time_ns() - t0_, bytes_);
	}

	nnc_perf_scope(const nnc_perf_scope&) = delete;
	nnc_perf_scope& operator=(const nnc_perf_scope&) = delete;

private:
	nnc_perf_slot slot_;
	uint64_t bytes_;
	int64_t t0_;
};
