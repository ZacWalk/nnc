// nnc runtime: arena, tensor graph, and forward dispatch for Gemma GGUF
// inference. Hot ops route to JIT/SIMD kernels in nn_ops.cpp; the rest
// are simple stride-aware reference loops.

#include "runtime.h"
#include "nn_ops.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>

#define NNC_MEM_ALIGN 16

// ============================================================================
// assert
// ============================================================================

[[noreturn]] void nnc_assert_fail(const char* expr, const char* file, const int line)
{
	fprintf(stderr, "\nnnc: ASSERT failed: %s\n  at %s:%d\n", expr, file, line);
	fflush(stderr);
	_exit(3);
}

// ============================================================================
// time
// ============================================================================

using nnc_clock = std::chrono::steady_clock;
static nnc_clock::time_point g_t0{};

void nnc_time_init()
{
	g_t0 = nnc_clock::now();
}

int64_t nnc_time_us()
{
	if (g_t0.time_since_epoch().count() == 0) nnc_time_init();
	return std::chrono::duration_cast<std::chrono::microseconds>(nnc_clock::now() - g_t0).count();
}

// ============================================================================
// type metadata
// ============================================================================

struct nnc_type_info
{
	int blck; // elements per storage block
	size_t bytes; // bytes per storage block
};

static constexpr nnc_type_info g_type_info[NNC_TYPE_COUNT] = {
	/* F32  */ {1, 4},
	/* F16  */ {1, 2},
	/* I32  */ {1, 4},
	/* BF16 */ {1, 2},
};

// AVX2 batched bf16 -> f32: 8 lanes/iter via vpmovzxwd + vpslld.
void nnc_bf16_to_f32_row(const nnc_bf16_t* src, float* dst, const size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
		const __m256i wide = _mm256_cvtepu16_epi32(lo);
		const __m256i shifted = _mm256_slli_epi32(wide, 16);
		_mm256_storeu_ps(dst + i, _mm256_castsi256_ps(shifted));
	}
	for (; i < n; ++i)
	{
		const uint32_t u = static_cast<uint32_t>(src[i]) << 16;
		float f;
		memcpy(&f, &u, 4);
		dst[i] = f;
	}
}

size_t nnc_type_size(const nnc_type t) { return g_type_info[t].bytes; }
int nnc_blck_size(const nnc_type t) { return g_type_info[t].blck; }

float nnc_type_sizef(const nnc_type t)
{
	return static_cast<float>(g_type_info[t].bytes) / static_cast<float>(g_type_info[t].blck);
}

size_t nnc_element_size(const nnc_tensor* t) { return g_type_info[t->type].bytes; }

int nnc_nelements(const nnc_tensor* t)
{
	return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

size_t nnc_nbytes(const nnc_tensor* t)
{
	const auto& ti = g_type_info[t->type];
	return static_cast<size_t>(nnc_nelements(t)) * ti.bytes / ti.blck;
}

void* nnc_get_data(const nnc_tensor* t) { return t->data; }

// ============================================================================
// arena
// ============================================================================

struct nnc_context
{
	size_t mem_size;
	void* mem_buffer;
	bool owns_buffer;
	size_t offset; // bump pointer in bytes
};

static size_t align_up(const size_t n, const size_t a) { return (n + a - 1) & ~(a - 1); }

static void* arena_alloc(nnc_context* ctx, const size_t bytes)
{
	const size_t off = align_up(ctx->offset, NNC_MEM_ALIGN);
	NNC_ASSERT(off + bytes <= ctx->mem_size && "nnc arena out of memory");
	ctx->offset = off + bytes;
	return static_cast<char*>(ctx->mem_buffer) + off;
}

nnc_context* nnc_init(const nnc_init_params params)
{
	auto* ctx = static_cast<nnc_context*>(malloc(sizeof(nnc_context)));
	if (!ctx) return nullptr;
	ctx->mem_size = params.mem_size;
	ctx->mem_buffer = params.mem_buffer;
	ctx->owns_buffer = false;
	if (!ctx->mem_buffer)
	{
		ctx->mem_buffer = malloc(params.mem_size);
		ctx->owns_buffer = true;
	}
	ctx->offset = 0;
	return ctx;
}

void nnc_free(nnc_context* ctx)
{
	if (!ctx) return;
	if (ctx->owns_buffer) free(ctx->mem_buffer);
	free(ctx);
}

size_t nnc_used_mem(const nnc_context* ctx) { return ctx->offset; }

// ============================================================================
// tensor builders
// ============================================================================

static nnc_tensor* alloc_tensor(nnc_context* ctx, const nnc_type t, const int n_dims, const int ne[NNC_MAX_DIMS],
                                void* data)
{
	auto* x = static_cast<nnc_tensor*>(arena_alloc(ctx, sizeof(nnc_tensor)));
	x->type = t;
	x->n_dims = n_dims;
	for (int i = 0; i < NNC_MAX_DIMS; ++i) x->ne[i] = ne[i];
	x->nb[0] = g_type_info[t].bytes;
	x->nb[1] = x->nb[0] * (ne[0] / g_type_info[t].blck);
	x->nb[2] = x->nb[1] * ne[1];
	x->nb[3] = x->nb[2] * ne[2];
	x->op = NNC_OP_NONE;
	x->src0 = nullptr;
	x->src1 = nullptr;
	x->op_params[0] = x->op_params[1] = x->op_params[2] = x->op_params[3] = 0;

	if (data)
	{
		x->data = data;
	}
	else
	{
		x->data = arena_alloc(ctx, nnc_nbytes(x));
	}
	return x;
}

static nnc_tensor* new_tensor(nnc_context* ctx, const nnc_type t, const int n_dims, const int ne0, const int ne1,
                              const int ne2, const int ne3)
{
	const int ne[NNC_MAX_DIMS] = {ne0, ne1, ne2, ne3};
	return alloc_tensor(ctx, t, n_dims, ne, nullptr);
}

nnc_tensor* nnc_new_tensor_1d(nnc_context* ctx, const nnc_type t, const int ne0)
{
	return new_tensor(ctx, t, 1, ne0, 1, 1, 1);
}

nnc_tensor* nnc_new_tensor_2d(nnc_context* ctx, const nnc_type t, const int ne0, const int ne1)
{
	return new_tensor(ctx, t, 2, ne0, ne1, 1, 1);
}

nnc_tensor* nnc_new_tensor_3d(nnc_context* ctx, const nnc_type t, const int ne0, const int ne1, const int ne2)
{
	return new_tensor(ctx, t, 3, ne0, ne1, ne2, 1);
}

nnc_tensor* nnc_new_f32(nnc_context* ctx, const float value)
{
	auto* x = nnc_new_tensor_1d(ctx, NNC_TYPE_F32, 1);
	*static_cast<float*>(x->data) = value;
	return x;
}

// ============================================================================
// op builders (just construct nodes; computation happens in graph_compute)
// ============================================================================

static nnc_tensor* dup_shape(nnc_context* ctx, const nnc_tensor* a)
{
	return alloc_tensor(ctx, a->type, a->n_dims, a->ne, nullptr);
}

nnc_tensor* nnc_add(nnc_context* ctx, nnc_tensor* a, nnc_tensor* b)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_ADD;
	r->src0 = a;
	r->src1 = b;
	return r;
}

nnc_tensor* nnc_mul(nnc_context* ctx, nnc_tensor* a, nnc_tensor* b)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_MUL;
	r->src0 = a;
	r->src1 = b;
	return r;
}

nnc_tensor* nnc_scale(nnc_context* ctx, nnc_tensor* a, nnc_tensor* s)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_SCALE;
	r->src0 = a;
	r->src1 = s;
	return r;
}

nnc_tensor* nnc_repeat(nnc_context* ctx, nnc_tensor* a, nnc_tensor* b)
{
	auto* r = alloc_tensor(ctx, a->type, b->n_dims, b->ne, nullptr);
	r->op = NNC_OP_REPEAT;
	r->src0 = a;
	r->src1 = b;
	return r;
}

nnc_tensor* nnc_norm(nnc_context* ctx, nnc_tensor* a)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_NORM;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_gelu(nnc_context* ctx, nnc_tensor* a)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_GELU;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_soft_max(nnc_context* ctx, nnc_tensor* a)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_SOFT_MAX;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_diag_mask_inf(nnc_context* ctx, nnc_tensor* a, const int n_past)
{
	auto* r = dup_shape(ctx, a);
	r->op = NNC_OP_DIAG_MASK_INF;
	r->src0 = a;
	r->op_params[0] = n_past;
	return r;
}

nnc_tensor* nnc_mul_mat(nnc_context* ctx, nnc_tensor* a, nnc_tensor* b)
{
	// out shape: [a->ne[1], b->ne[1], a->ne[2], b->ne[3]]
	const int ne_out[NNC_MAX_DIMS] = {a->ne[1], b->ne[1], a->ne[2], b->ne[3]};
	const int n_dims = (a->n_dims > b->n_dims ? a->n_dims : b->n_dims);
	auto* r = alloc_tensor(ctx, NNC_TYPE_F32, n_dims, ne_out, nullptr);
	r->op = NNC_OP_MUL_MAT;
	r->src0 = a;
	r->src1 = b;
	return r;
}

nnc_tensor* nnc_get_rows(nnc_context* ctx, nnc_tensor* a, nnc_tensor* idx)
{
	const int ne_out[NNC_MAX_DIMS] = {a->ne[0], idx->ne[0], 1, 1};
	auto* r = alloc_tensor(ctx, NNC_TYPE_F32, 2, ne_out, nullptr);
	r->op = NNC_OP_GET_ROWS;
	r->src0 = a;
	r->src1 = idx;
	return r;
}

nnc_tensor* nnc_cpy(nnc_context* ctx, nnc_tensor* a, nnc_tensor* dst)
{
	// Result shares dst's storage so downstream consumers see the destination.
	auto* r = alloc_tensor(ctx, dst->type, dst->n_dims, dst->ne, dst->data);
	r->op = NNC_OP_CPY;
	r->src0 = a;
	r->src1 = dst;
	return r;
}

nnc_tensor* nnc_view_1d(nnc_context* ctx, nnc_tensor* a, const int ne0, const size_t offset)
{
	const int ne_v[NNC_MAX_DIMS] = {ne0, 1, 1, 1};
	auto* r = alloc_tensor(ctx, a->type, 1, ne_v, static_cast<char*>(a->data) + offset);
	r->op = NNC_OP_VIEW;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_view_2d(nnc_context* ctx, nnc_tensor* a, const int ne0, const int ne1, const size_t nb1,
                        const size_t offset)
{
	const int ne_v[NNC_MAX_DIMS] = {ne0, ne1, 1, 1};
	auto* r = alloc_tensor(ctx, a->type, 2, ne_v, static_cast<char*>(a->data) + offset);
	r->nb[1] = nb1;
	r->nb[2] = nb1 * ne1;
	r->nb[3] = r->nb[2];
	r->op = NNC_OP_VIEW;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_reshape_3d(nnc_context* ctx, nnc_tensor* a, const int ne0, const int ne1, const int ne2)
{
	NNC_ASSERT(ne0 * ne1 * ne2 == nnc_nelements(a));
	const int ne_r[NNC_MAX_DIMS] = {ne0, ne1, ne2, 1};
	auto* r = alloc_tensor(ctx, a->type, 3, ne_r, a->data);
	r->op = NNC_OP_RESHAPE;
	r->src0 = a;
	return r;
}

nnc_tensor* nnc_permute(nnc_context* ctx, nnc_tensor* a, const int axis0, const int axis1, const int axis2,
                        const int axis3)
{
	auto* r = alloc_tensor(ctx, a->type, a->n_dims, a->ne, a->data);
	const int axes[4] = {axis0, axis1, axis2, axis3};
	int ne_p[4];
	size_t nb_p[4];

	for (int i = 0; i < 4; ++i)
	{
		ne_p[axes[i]] = a->ne[i];
		nb_p[axes[i]] = a->nb[i];
	}
	for (int i = 0; i < 4; ++i)
	{
		r->ne[i] = ne_p[i];
		r->nb[i] = nb_p[i];
	}
	r->op = NNC_OP_PERMUTE;
	r->src0 = a;
	return r;
}

// ============================================================================
// graph build (DFS topo)
// ============================================================================

static bool graph_contains(const nnc_cgraph* g, const nnc_tensor* t)
{
	for (int i = 0; i < g->n_nodes; ++i)
		if (g->nodes[i] == t) return true;
	return false;
}

static void visit(nnc_cgraph* g, nnc_tensor* t)
{
	if (!t || graph_contains(g, t)) return;
	if (t->src0) visit(g, t->src0);
	if (t->src1) visit(g, t->src1);
	NNC_ASSERT(g->n_nodes < NNC_MAX_NODES);
	g->nodes[g->n_nodes++] = t;
}

void nnc_build_forward_expand(nnc_cgraph* g, nnc_tensor* root) { visit(g, root); }

// ============================================================================
// FP16 <-> FP32 (F16C)
// ============================================================================

static inline float fp16_to_fp32(const nnc_fp16_t h)
{
	return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

static inline nnc_fp16_t fp32_to_fp16(const float f)
{
	const __m128i p = _mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT);
	return static_cast<nnc_fp16_t>(_mm_extract_epi16(p, 0));
}

// ============================================================================
// vec helpers (AVX2)
// ============================================================================

static inline void vec_add_f32(const size_t n, float* y, const float* a, const float* b)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m256 va = _mm256_loadu_ps(a + i);
		const __m256 vb = _mm256_loadu_ps(b + i);
		_mm256_storeu_ps(y + i, _mm256_add_ps(va, vb));
	}
	for (; i < n; ++i) y[i] = a[i] + b[i];
}

static inline void vec_mul_f32(const size_t n, float* y, const float* a, const float* b)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m256 va = _mm256_loadu_ps(a + i);
		const __m256 vb = _mm256_loadu_ps(b + i);
		_mm256_storeu_ps(y + i, _mm256_mul_ps(va, vb));
	}
	for (; i < n; ++i) y[i] = a[i] * b[i];
}

static inline void vec_scale_f32(const size_t n, float* y, const float* x, const float s)
{
	const __m256 vs = _mm256_set1_ps(s);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
	}
	for (; i < n; ++i) y[i] = x[i] * s;
}

static inline float vec_dot_f32(const size_t n, const float* a, const float* b)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		const __m256 va = _mm256_loadu_ps(a + i);
		const __m256 vb = _mm256_loadu_ps(b + i);
		acc = _mm256_fmadd_ps(va, vb, acc);
	}
	float buf[8];
	_mm256_storeu_ps(buf, acc);
	float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
	for (; i < n; ++i) s += a[i] * b[i];
	return s;
}

// ============================================================================
// forward implementations
// ============================================================================

static void forward_add(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const auto* b = dst->src1;
	NNC_ASSERT(a->type == NNC_TYPE_F32 && b->type == NNC_TYPE_F32 && dst->type == NNC_TYPE_F32);
	const size_t n = nnc_nelements(dst);
	vec_add_f32(n, static_cast<float*>(dst->data), static_cast<const float*>(a->data),
	            static_cast<const float*>(b->data));
}

static void forward_mul(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const auto* b = dst->src1;
	NNC_ASSERT(a->type == NNC_TYPE_F32 && b->type == NNC_TYPE_F32);
	const size_t n = nnc_nelements(dst);
	vec_mul_f32(n, static_cast<float*>(dst->data), static_cast<const float*>(a->data),
	            static_cast<const float*>(b->data));
}

static void forward_scale(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const auto* s = dst->src1;
	const float scalar = *static_cast<const float*>(s->data);
	const size_t n = nnc_nelements(dst);
	vec_scale_f32(n, static_cast<float*>(dst->data), static_cast<const float*>(a->data), scalar);
}

static void forward_gelu(const nnc_tensor* dst)
{
	const size_t n = nnc_nelements(dst);
	nnc_gelu_f32(static_cast<float*>(dst->data), static_cast<const float*>(dst->src0->data), n);
}

static void forward_norm(const nnc_tensor* dst)
{
	// per-row layernorm over ne[0]; rows = ne[1]*ne[2]*ne[3]. 
	const auto* a = dst->src0;
	const int ne0 = a->ne[0];
	const int rows = a->ne[1] * a->ne[2] * a->ne[3];
	constexpr float eps = 1e-5f;
	for (int r = 0; r < rows; ++r)
	{
		const auto x = (const float*)(static_cast<const char*>(a->data) + static_cast<size_t>(r) * a->nb[1]);
		const auto y = (float*)(static_cast<char*>(dst->data) + static_cast<size_t>(r) * dst->nb[1]);
		nnc_layernorm_f32(y, x, static_cast<size_t>(ne0), eps);
	}
}

static void forward_soft_max(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const int ne0 = a->ne[0];
	const int rows = a->ne[1] * a->ne[2] * a->ne[3];
	for (int r = 0; r < rows; ++r)
	{
		const auto x = (const float*)(static_cast<const char*>(a->data) + static_cast<size_t>(r) * a->nb[1]);
		const auto y = (float*)(static_cast<char*>(dst->data) + static_cast<size_t>(r) * dst->nb[1]);
		if (y != x) memcpy(y, x, sizeof(float) * static_cast<size_t>(ne0));
		nnc_softmax_f32_inplace(y, static_cast<size_t>(ne0));
	}
}

static void forward_diag_mask_inf(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const int n_past = dst->op_params[0];
	const int ne0 = a->ne[0]; // n_past + N
	const int ne1 = a->ne[1]; // N
	const int ne2 = a->ne[2]; // n_head
	if (dst->data != a->data) memcpy(dst->data, a->data, nnc_nbytes(a));
	constexpr float ninf = -INFINITY;
	for (int k = 0; k < ne2; ++k)
	{
		for (int j = 0; j < ne1; ++j)
		{
			const auto row = (float*)(static_cast<char*>(dst->data) + static_cast<size_t>(k) * dst->nb[2] + static_cast<
				size_t>(j) * dst->nb[1]);
			const int allowed = n_past + j + 1;
			for (int i = allowed; i < ne0; ++i) row[i] = ninf;
		}
	}
}

static void forward_repeat(const nnc_tensor* dst)
{
	// general broadcast: each output index is taken modulo the source extents.
	const auto* a = dst->src0;
	NNC_ASSERT(a->type == NNC_TYPE_F32 && dst->type == NNC_TYPE_F32);
	const int ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
	const int sa0 = a->ne[0], sa1 = a->ne[1], sa2 = a->ne[2], sa3 = a->ne[3];
	for (int i3 = 0; i3 < ne3; ++i3)
		for (int i2 = 0; i2 < ne2; ++i2)
			for (int i1 = 0; i1 < ne1; ++i1)
			{
				const auto y = (float*)(static_cast<char*>(dst->data) + static_cast<size_t>(i3) * dst->nb[3] +
					static_cast<
						size_t>(i2) * dst->nb[2] + static_cast<size_t>(i1) * dst->nb[1]);
				const auto x = (const float*)(static_cast<const char*>(a->data) + static_cast<size_t>(i3 % sa3) * a->nb[
						3] +
					static_cast<size_t>(i2 % sa2) * a->nb[2] + static_cast<size_t>(i1 % sa1) * a->nb[1]);
				for (int i0 = 0; i0 < ne0; ++i0) y[i0] = x[i0 % sa0];
			}
}

static void forward_get_rows(const nnc_tensor* dst)
{
	const auto* a = dst->src0;
	const auto* idx = dst->src1;
	NNC_ASSERT(idx->type == NNC_TYPE_I32);
	NNC_ASSERT(dst->type == NNC_TYPE_F32);
	const int ne0 = a->ne[0];
	const int rows = idx->ne[0];
	const auto* indices = static_cast<const int32_t*>(idx->data);
	for (int r = 0; r < rows; ++r)
	{
		const auto y = (float*)(static_cast<char*>(dst->data) + static_cast<size_t>(r) * dst->nb[1]);
		const void* xrow = static_cast<const char*>(a->data) + static_cast<size_t>(indices[r]) * a->nb[1];
		if (a->type == NNC_TYPE_F32)
		{
			memcpy(y, xrow, sizeof(float) * static_cast<size_t>(ne0));
		}
		else if (a->type == NNC_TYPE_F16)
		{
			const auto* h = static_cast<const nnc_fp16_t*>(xrow);
			for (int i = 0; i < ne0; ++i) y[i] = fp16_to_fp32(h[i]);
		}
		else
		{
			NNC_ASSERT(!"get_rows: unsupported source type");
		}
	}
}

static void forward_cpy(const nnc_tensor* dst)
{
	// dst's data already aliases the destination tensor's buffer (set at
	// build time). 
	const auto* a = dst->src0;
	const int ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
	NNC_ASSERT(static_cast<size_t>(ne0) * ne1 * ne2 * ne3 == static_cast<size_t>(nnc_nelements(dst)));

	const size_t dst_es = nnc_element_size(dst);
	const auto dp_base = static_cast<char*>(dst->data);
	size_t dst_off = 0;

	for (int i3 = 0; i3 < ne3; ++i3)
		for (int i2 = 0; i2 < ne2; ++i2)
			for (int i1 = 0; i1 < ne1; ++i1)
			{
				const char* ap = static_cast<const char*>(a->data)
					+ static_cast<size_t>(i3) * a->nb[3] + static_cast<size_t>(i2) * a->nb[2] + static_cast<size_t>(i1)
					* a->nb[1];

				if (a->type == dst->type && a->nb[0] == nnc_element_size(a))
				{
					memcpy(dp_base + dst_off, ap, dst_es * static_cast<size_t>(ne0));
				}
				else if (a->type == NNC_TYPE_F32 && dst->type == NNC_TYPE_F32)
				{
					auto* dq = (float*)(dp_base + dst_off);
					for (int i0 = 0; i0 < ne0; ++i0)
						dq[i0] = *(const float*)(ap + static_cast<size_t>(i0) * a->nb[0]);
				}
				else if (a->type == NNC_TYPE_F32 && dst->type == NNC_TYPE_F16)
				{
					auto* dq = (nnc_fp16_t*)(dp_base + dst_off);
					for (int i0 = 0; i0 < ne0; ++i0)
						dq[i0] = fp32_to_fp16(*(const float*)(ap + static_cast<size_t>(i0) * a->nb[0]));
				}
				else if (a->type == NNC_TYPE_F16 && dst->type == NNC_TYPE_F32)
				{
					auto* dq = (float*)(dp_base + dst_off);
					for (int i0 = 0; i0 < ne0; ++i0)
						dq[i0] = fp16_to_fp32(*(const nnc_fp16_t*)(ap + static_cast<size_t>(i0) * a->nb[0]));
				}
				else
				{
					NNC_ASSERT(!"cpy: unsupported type combo");
				}

				dst_off += dst_es * static_cast<size_t>(ne0);
			}
}

// --- mul_mat ---------------------------------------------------------------

// FP16 weights * FP32 activations -> FP32. W is [in, out] contiguous (nb00==2,
// nb01 == 2*in). x is [in, N], may have nb10 != 4 (we do NOT support that
// here -- main.cpp keeps x contiguous for the W*x calls).
static void forward_mul_mat_f16_f32_gemv(const nnc_tensor* dst)
{
	const auto* W = dst->src0;
	const auto* X = dst->src1;
	NNC_ASSERT(W->n_dims == 2);
	NNC_ASSERT(W->nb[0] == sizeof(nnc_fp16_t));
	NNC_ASSERT(X->nb[0] == sizeof(float));

	const int rows = W->ne[1]; // output rows
	const int cols = W->ne[0]; // inner / input dim
	const int ncols = X->ne[1]; // number of x columns (== N)

	const float* bias = nnc_fused_bias_for(dst);
	void* dst_buffer = nnc_fused_dst_for(dst);
	const bool fuse_gelu = nnc_fused_gelu_for(dst);
	if (!dst_buffer) dst_buffer = dst->data;

	for (int c = 0; c < ncols; ++c)
	{
		const auto xcol = (const float*)(static_cast<const char*>(X->data) + static_cast<size_t>(c) * X->nb[1]);
		const auto ycol = (float*)(static_cast<char*>(dst_buffer) + static_cast<size_t>(c) * dst->nb[1]);
		nnc_gemv_f16w_f32x(W->data, xcol, ycol, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols));
		if (bias) nnc_add_inplace_f32(ycol, bias, static_cast<size_t>(rows));
		if (fuse_gelu) nnc_gelu_f32(ycol, ycol, static_cast<size_t>(rows));
	}
}

// Stride-aware F16 * F32 -> F32 for the attention K * Q case where A is the
// permuted FP16 KV cache and B is the FP32 query/score matrix.
static void forward_mul_mat_f16_f32_general(const nnc_tensor* dst)
{
	const auto* A = dst->src0;
	const auto* B = dst->src1;
	NNC_ASSERT(A->nb[0] == sizeof(nnc_fp16_t));
	NNC_ASSERT(B->nb[0] == sizeof(float));

	const int ne00 = A->ne[0];
	const int ne01 = A->ne[1];
	const int ne02 = A->ne[2];
	const int ne03 = A->ne[3];
	const int ne11 = B->ne[1];

	for (int i3 = 0; i3 < ne03; ++i3)
		for (int i2 = 0; i2 < ne02; ++i2)
			for (int ir = 0; ir < ne01; ++ir)
			{
				const char* arow = static_cast<const char*>(A->data)
					+ static_cast<size_t>(i3) * A->nb[3] + static_cast<size_t>(i2) * A->nb[2] + static_cast<size_t>(ir)
					* A->nb[1];
				for (int ic = 0; ic < ne11; ++ic)
				{
					const char* bcol = static_cast<const char*>(B->data)
						+ static_cast<size_t>(i3) * B->nb[3] + static_cast<size_t>(i2) * B->nb[2] + static_cast<size_t>(
							ic) * B->nb[1];
					const auto yp = (float*)(static_cast<char*>(dst->data)
						+ static_cast<size_t>(i3) * dst->nb[3] + static_cast<size_t>(i2) * dst->nb[2] + static_cast<
							size_t>(ic) * dst->nb[1] + static_cast<size_t>(ir) * dst->nb[0]);
					double s = 0.0;
					for (int k = 0; k < ne00; ++k)
					{
						const auto h = *(const nnc_fp16_t*)(arow + static_cast<size_t>(k) * A->nb[0]);
						const auto x = *(const float*)(bcol + static_cast<size_t>(k) * B->nb[0]);
						s += static_cast<double>(fp16_to_fp32(h)) * static_cast<double>(x);
					}
					*yp = static_cast<float>(s);
				}
			}
}

// FP32 weights * FP32 activations -> FP32 (used for attention K*Q and V*S).
// Handles arbitrary nb strides on src0/src1 (post-permute), including the
// transposed-V case where src0->nb[0] != sizeof(float).
static void forward_mul_mat_f32_f32(const nnc_tensor* dst)
{
	const auto* A = dst->src0;
	const auto* B = dst->src1;
	NNC_ASSERT(B->nb[0] == sizeof(float));

	const int ne00 = A->ne[0]; // inner
	const int ne01 = A->ne[1]; // out rows
	const int ne02 = A->ne[2];
	const int ne03 = A->ne[3];
	const int ne11 = B->ne[1]; // out cols

	const bool a_contig = (A->nb[0] == sizeof(float));

	for (int i3 = 0; i3 < ne03; ++i3)
		for (int i2 = 0; i2 < ne02; ++i2)
			for (int ir = 0; ir < ne01; ++ir)
			{
				const char* arow = static_cast<const char*>(A->data)
					+ static_cast<size_t>(i3) * A->nb[3] + static_cast<size_t>(i2) * A->nb[2] + static_cast<size_t>(ir)
					* A->nb[1];
				for (int ic = 0; ic < ne11; ++ic)
				{
					const auto bcol = (const float*)(static_cast<const char*>(B->data)
						+ static_cast<size_t>(i3) * B->nb[3] + static_cast<size_t>(i2) * B->nb[2] + static_cast<size_t>(
							ic) * B->nb[1]);
					const auto yp = (float*)(static_cast<char*>(dst->data)
						+ static_cast<size_t>(i3) * dst->nb[3] + static_cast<size_t>(i2) * dst->nb[2] + static_cast<
							size_t>(ic) * dst->nb[1] + static_cast<size_t>(ir) * dst->nb[0]);
					if (a_contig)
					{
						*yp = vec_dot_f32(static_cast<size_t>(ne00), (const float*)arow, bcol);
					}
					else
					{
						double s = 0.0;
						for (int k = 0; k < ne00; ++k)
							s += static_cast<double>(*(const float*)(arow + static_cast<size_t>(k) * A->nb[0])) *
								static_cast<double>
								(bcol[k]);
						*yp = static_cast<float>(s);
					}
				}
			}
}

static void forward_mul_mat(const nnc_tensor* dst)
{
	const nnc_type t0 = dst->src0->type;
	const nnc_type t1 = dst->src1->type;
	if (t0 == NNC_TYPE_F16 && t1 == NNC_TYPE_F32)
	{
		// 2D contiguous FP16 weights -> use the JITted gemv fast path.
		// Permuted / 3D KV cache -> stride-aware general path.
		const auto* A = dst->src0;
		if (A->n_dims == 2 && A->nb[1] == A->nb[0] * static_cast<size_t>(A->ne[0]))
			forward_mul_mat_f16_f32_gemv(dst);
		else
			forward_mul_mat_f16_f32_general(dst);
	}
	else if (t0 == NNC_TYPE_F32 && t1 == NNC_TYPE_F32)
	{
		forward_mul_mat_f32_f32(dst);
	}
	else
	{
		NNC_ASSERT(!"mul_mat: unsupported type combo");
	}
}

// ============================================================================
// graph_compute
// ============================================================================

void nnc_graph_compute(nnc_context* /*ctx*/, const nnc_cgraph* g)
{
	nnc_graph_prefuse(g);

	for (int i = 0; i < g->n_nodes; ++i)
	{
		nnc_tensor* t = g->nodes[i];
		switch (t->op)
		{
		case NNC_OP_NONE:
		case NNC_OP_VIEW:
		case NNC_OP_RESHAPE:
		case NNC_OP_PERMUTE:
			break;
		case NNC_OP_ADD: forward_add(t);
			break;
		case NNC_OP_MUL: forward_mul(t);
			break;
		case NNC_OP_SCALE: forward_scale(t);
			break;
		case NNC_OP_GELU: forward_gelu(t);
			break;
		case NNC_OP_NORM: forward_norm(t);
			break;
		case NNC_OP_SOFT_MAX: forward_soft_max(t);
			break;
		case NNC_OP_DIAG_MASK_INF: forward_diag_mask_inf(t);
			break;
		case NNC_OP_REPEAT: forward_repeat(t);
			break;
		case NNC_OP_GET_ROWS: forward_get_rows(t);
			break;
		case NNC_OP_CPY: forward_cpy(t);
			break;
		case NNC_OP_MUL_MAT: forward_mul_mat(t);
			break;
		default:
			fprintf(stderr, "nnc_graph_compute: unhandled op %d\n", static_cast<int>(t->op));
			NNC_ASSERT(!"unhandled op");
		}
	}
}
