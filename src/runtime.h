// nnc runtime: arena, tensor graph, and forward dispatch for the Gemma
// GGUF inference path.

#pragma once

#include <cstdint>
#include <cstring>

#define NNC_MAX_DIMS  4
#define NNC_MAX_NODES 4096

#ifndef NNC_ASSERT
// Custom assert: print "ASSERT failed: <expr> at <file>:<line>" to stderr,
// flush, and abort the process. No modal message box.
[[noreturn]] void nnc_assert_fail(const char* expr, const char* file, int line);
#define NNC_ASSERT(x) ((x) ? (void)0 : nnc_assert_fail(#x, __FILE__, __LINE__))
#endif

using nnc_fp16_t = uint16_t;
using nnc_bf16_t = uint16_t;

enum nnc_type
{
	NNC_TYPE_F32 = 0,
	NNC_TYPE_F16 = 1,
	NNC_TYPE_I32 = 2,
	NNC_TYPE_BF16 = 3,
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

// AVX2 batched bf16 -> f32 conversion (n must be multiple of 8).
void nnc_bf16_to_f32_row(const nnc_bf16_t* src, float* dst, size_t n);

enum nnc_op
{
	NNC_OP_NONE = 0,
	NNC_OP_ADD,
	NNC_OP_MUL,
	NNC_OP_REPEAT,
	NNC_OP_GELU,
	NNC_OP_NORM,
	NNC_OP_MUL_MAT,
	NNC_OP_SCALE,
	NNC_OP_CPY,
	NNC_OP_RESHAPE,
	NNC_OP_VIEW,
	NNC_OP_PERMUTE,
	NNC_OP_GET_ROWS,
	NNC_OP_DIAG_MASK_INF,
	NNC_OP_SOFT_MAX,
	NNC_OP_COUNT,
};

struct nnc_tensor
{
	nnc_type type;
	int n_dims;
	int ne[NNC_MAX_DIMS]; // shape
	size_t nb[NNC_MAX_DIMS]; // strides in bytes
	nnc_op op;
	struct nnc_tensor* src0;
	struct nnc_tensor* src1;
	void* data;
	int32_t op_params[4]; // small per-op scratch (e.g. n_past)
};

struct nnc_cgraph
{
	int n_nodes;
	struct nnc_tensor* nodes[NNC_MAX_NODES];
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
size_t nnc_used_mem(const struct nnc_context* ctx);

// --- type metadata ---
size_t nnc_type_size(nnc_type t);
float nnc_type_sizef(nnc_type t);
int nnc_blck_size(nnc_type t);
size_t nnc_element_size(const struct nnc_tensor* t);
size_t nnc_nbytes(const struct nnc_tensor* t);
int nnc_nelements(const struct nnc_tensor* t);
void* nnc_get_data(const struct nnc_tensor* t);

// --- tensor builders (allocate into the arena) ---
struct nnc_tensor* nnc_new_tensor_1d(struct nnc_context* ctx, nnc_type t, int ne0);
struct nnc_tensor* nnc_new_tensor_2d(struct nnc_context* ctx, nnc_type t, int ne0, int ne1);
struct nnc_tensor* nnc_new_tensor_3d(struct nnc_context* ctx, nnc_type t, int ne0, int ne1, int ne2);
struct nnc_tensor* nnc_new_f32(struct nnc_context* ctx, float value);

// --- op builders (graph nodes; computation is deferred to graph_compute) ---
struct nnc_tensor* nnc_add(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* b);
struct nnc_tensor* nnc_mul(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* b);
struct nnc_tensor* nnc_mul_mat(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* b);
struct nnc_tensor* nnc_scale(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* s);
struct nnc_tensor* nnc_repeat(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* b);
struct nnc_tensor* nnc_norm(struct nnc_context* ctx, struct nnc_tensor* a);
struct nnc_tensor* nnc_gelu(struct nnc_context* ctx, struct nnc_tensor* a);
struct nnc_tensor* nnc_soft_max(struct nnc_context* ctx, struct nnc_tensor* a);
struct nnc_tensor* nnc_diag_mask_inf(struct nnc_context* ctx, struct nnc_tensor* a, int n_past);
struct nnc_tensor* nnc_get_rows(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* idx);
struct nnc_tensor* nnc_cpy(struct nnc_context* ctx, struct nnc_tensor* a, struct nnc_tensor* dst);
struct nnc_tensor* nnc_view_1d(struct nnc_context* ctx, struct nnc_tensor* a, int ne0, size_t offset);
struct nnc_tensor* nnc_view_2d(struct nnc_context* ctx, struct nnc_tensor* a, int ne0, int ne1, size_t nb1,
                               size_t offset);
struct nnc_tensor* nnc_reshape_3d(struct nnc_context* ctx, struct nnc_tensor* a, int ne0, int ne1, int ne2);
struct nnc_tensor* nnc_permute(struct nnc_context* ctx, struct nnc_tensor* a, int axis0, int axis1, int axis2,
                               int axis3);

// --- graph build + run ---
void nnc_build_forward_expand(struct nnc_cgraph* g, struct nnc_tensor* root);
void nnc_graph_compute(struct nnc_context* ctx, const struct nnc_cgraph* g);

// --- timing helpers (microseconds since process start) ---
void nnc_time_init();
int64_t nnc_time_us();
