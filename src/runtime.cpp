// nnc runtime: arena allocator, dtype metadata, and tensor descriptors
// for the Gemma GGUF inference path. Computation lives in nn_ops.cpp
// (SIMD) and jit_ops.cpp (JIT); gemma.cpp drives those kernels directly.

#include "runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

#define NNC_MEM_ALIGN 16

// ============================================================================
// assert
// ============================================================================

[[noreturn]] void nnc_assert_fail(const char* expr, const char* file, const int line)
{
	fprintf(stderr, "\nnnc: ASSERT failed: %s\n  at %s:%d\n", expr, file, line);
	fflush(stderr);
	std::_Exit(3);
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

int64_t nnc_time_ns()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		nnc_clock::now().time_since_epoch()).count();
}

// ============================================================================
// phase profiler
// ============================================================================

bool g_nnc_perf_on = false;

namespace
{
	struct perf_slot_data
	{
		int64_t ns;
		uint64_t bytes;
		uint64_t calls;
	};

	perf_slot_data g_perf[NNC_PERF_COUNT]{};

	const char* const g_perf_names[NNC_PERF_COUNT] = {
		"embed",
		"ple_prep",
		"norm",
		"attn_qkv",
		"attn_rope",
		"attn_core",
		"attn_out",
		"ffn_gate_up",
		"ffn_act",
		"ffn_down",
		"ple_layer",
		"lm_head",
	};
}

void nnc_perf_enable(const bool on) { g_nnc_perf_on = on; }

void nnc_perf_reset() { memset(g_perf, 0, sizeof(g_perf)); }

void nnc_perf_add(const nnc_perf_slot slot, const int64_t ns, const uint64_t bytes)
{
	auto& s = g_perf[slot];
	s.ns += ns;
	s.bytes += bytes;
	s.calls += 1;
}

void nnc_perf_report(const char* title, const int n_tokens)
{
	int64_t total_ns = 0;
	uint64_t total_bytes = 0;
	for (const auto& s : g_perf)
	{
		total_ns += s.ns;
		total_bytes += s.bytes;
	}
	if (total_ns <= 0)
	{
		printf("\nnnc perf (%s): no samples\n", title);
		return;
	}

	printf("\nnnc perf — %s\n", title);
	printf("  %-12s %10s %7s %10s %10s %10s\n",
	       "phase", "ms", "%", "calls", "us/call", "GB/s");
	printf("  %-12s %10s %7s %10s %10s %10s\n",
	       "------------", "----------", "-------", "----------", "----------", "----------");
	for (int i = 0; i < NNC_PERF_COUNT; ++i)
	{
		const auto& s = g_perf[i];
		if (s.calls == 0) continue;
		const double ms = static_cast<double>(s.ns) / 1e6;
		const double pct = 100.0 * static_cast<double>(s.ns) / static_cast<double>(total_ns);
		const double us_per_call = static_cast<double>(s.ns) / 1e3 / static_cast<double>(s.calls);
		printf("  %-12s %10.1f %6.1f%% %10llu %10.1f",
		       g_perf_names[i], ms, pct,
		       static_cast<unsigned long long>(s.calls), us_per_call);
		if (s.bytes > 0)
			printf(" %10.1f", static_cast<double>(s.bytes) / static_cast<double>(s.ns));
		printf("\n");
	}
	printf("  %-12s %10.1f %6.1f%%\n", "TOTAL",
	       static_cast<double>(total_ns) / 1e6, 100.0);
	if (total_bytes > 0)
	{
		printf("  weight traffic: %.2f GB @ %.1f GB/s effective\n",
		       static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0),
		       static_cast<double>(total_bytes) / static_cast<double>(total_ns));
	}
	if (n_tokens > 0)
	{
		printf("  %.2f ms/token over %d tokens\n",
		       static_cast<double>(total_ns) / 1e6 / n_tokens, n_tokens);
	}
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
	// Q8_0 split: 32 elements per logical block. Per-element bytes is
	// 1 (the int8 qs) + 2/32 (the bf16 scale) = 1.0625, but type_size is
	// integer-only, so report bytes-per-block = 34 (32*1 + 2) which is
	// what gets used when a caller wants to size a Q8 buffer.
	/* Q8_0 */ {32, 34},
	// Q4 split: 32 elements -> 16 bytes of nibbles + one bf16 scale +
	// one bf16 bias = 20 bytes per block (0.625 bytes/weight).
	/* Q4_S */ {32, 20},
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

int64_t nnc_nelements(const nnc_tensor* t)
{
	return static_cast<int64_t>(t->ne[0])
		* static_cast<int64_t>(t->ne[1])
		* static_cast<int64_t>(t->ne[2])
		* static_cast<int64_t>(t->ne[3]);
}

size_t nnc_nbytes(const nnc_tensor* t)
{
	const auto& ti = g_type_info[t->type];
	return static_cast<size_t>(nnc_nelements(t)) * ti.bytes / ti.blck;
}

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

static size_t align_up(const size_t n, const size_t a)
{
	// a is always a power of two; guard against overflow when n is near
	// SIZE_MAX so the bump-pointer math below stays well-defined.
	NNC_ASSERT(a > 0 && (a & (a - 1)) == 0);
	NNC_ASSERT(n <= SIZE_MAX - (a - 1));
	return (n + a - 1) & ~(a - 1);
}

static void* arena_alloc(nnc_context* ctx, const size_t bytes)
{
	const size_t off = align_up(ctx->offset, NNC_MEM_ALIGN);
	// Use "off <= mem_size - bytes" so the addition can never wrap.
	NNC_ASSERT(bytes <= ctx->mem_size && off <= ctx->mem_size - bytes
		&& "nnc arena out of memory");
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
		if (!ctx->mem_buffer)
		{
			free(ctx);
			return nullptr;
		}
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

// ============================================================================
// tensor descriptors
// ============================================================================

nnc_tensor* nnc_new_tensor_1d(nnc_context* ctx, const nnc_type t, const int ne0)
{
	auto* x = static_cast<nnc_tensor*>(arena_alloc(ctx, sizeof(nnc_tensor)));
	x->type = t;
	x->n_dims = 1;
	x->ne[0] = ne0;
	x->ne[1] = x->ne[2] = x->ne[3] = 1;
	x->nb[0] = g_type_info[t].bytes;
	x->nb[1] = x->nb[0] * (ne0 / g_type_info[t].blck);
	x->nb[2] = x->nb[1];
	x->nb[3] = x->nb[2];
	x->data = arena_alloc(ctx, nnc_nbytes(x));
	return x;
}
