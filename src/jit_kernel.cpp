// nnc — CPU feature detection and the JIT kernel cache.
// Detects AVX2/FMA/F16C via cpuid at startup, refuses to run on older
// CPUs, and owns the (op, dtype, shape)-keyed cache that JITs each kernel
// once and hands out reusable typed function pointers.

#include "jit_kernel.h"

#include "jit_buffer.h"
#include "jit_ops.h"
#include "sys.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

// ---- CPU detection -----------------------------------------------------

static cpu_features detect()
{
	cpu_features f{};

	int regs[4] = {0, 0, 0, 0};

	sys_cpuid(regs, 1);
	const bool osxsave = (regs[2] & (1 << 27)) != 0;
	const bool avx = (regs[2] & (1 << 28)) != 0;
	f.fma = (regs[2] & (1 << 12)) != 0;
	f.f16c = (regs[2] & (1 << 29)) != 0;

	sys_cpuidex(regs, 7, 0);
	const bool avx2_bit = (regs[1] & (1 << 5)) != 0;

	bool ymm_enabled = false;
	if (osxsave && avx)
	{
		const uint64_t xcr0 = sys_xgetbv(0);
		ymm_enabled = (xcr0 & 0x6) == 0x6;
	}

	f.avx2 = avx2_bit && ymm_enabled;
	return f;
}

const cpu_features& nnc_cpu_features()
{
	static const cpu_features f = detect();
	return f;
}

void nnc_require_avx2_fma()
{
	const auto& f = nnc_cpu_features();
	if (!f.avx2 || !f.fma)
	{
		std::fprintf(stderr,
		             "nnc: this CPU is not supported (avx2=%d fma=%d). "
		             "nnc requires AVX2 + FMA.\n",
		             f.avx2 ? 1 : 0, f.fma ? 1 : 0);
		std::exit(2);
	}
}

// ---- Kernel cache ------------------------------------------------------

struct jit_kernel_cache::impl
{
	struct entry
	{
		std::unique_ptr<jit_buffer> buf;
		void* fn = nullptr;
	};

	// Key packs (rows, cols) into a single 64-bit value.
	std::unordered_map<uint64_t, entry> gemv_;
	std::unordered_map<uint32_t, entry> dot_f16_;
	std::unordered_map<uint64_t, entry> gemv_f16w_f32x_;
	std::unordered_map<uint64_t, entry> gemv_bf16w_f32x_;
	std::unordered_map<uint32_t, entry> gemv_q8_0_1row_;
	// shared_mutex: hot-path lookups take a shared lock (concurrent
	// readers OK), and only first-time codegen takes the exclusive lock.
	// After first-token warmup every gemv hits the shared path. The maps
	// are append-only; once a slot is inserted its `fn` pointer never
	// changes, so reading it under a shared lock is safe.
	mutable std::shared_mutex mu_;
};

jit_kernel_cache::jit_kernel_cache() : p_(std::make_unique<impl>())
{
}

jit_kernel_cache::~jit_kernel_cache() = default;

static inline uint64_t pack(const uint32_t a, const uint32_t b)
{
	return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

// Generic shared/exclusive lookup helper. The shared (read) path is the
// steady-state hit; the exclusive (write) path runs once per shape on
// first touch. We re-check the map under the exclusive lock so two
// threads racing the same key only build the kernel once.
template <typename Map, typename Key, typename Build>
static void* lookup_or_build(std::shared_mutex& mu, Map& m, const Key& key, Build build)
{
	{
		std::shared_lock<std::shared_mutex> rk(mu);
		const auto it = m.find(key);
		if (it != m.end()) return it->second.fn;
	}
	auto buf = std::make_unique<jit_buffer>();
	build(*buf);
	void* fn = buf->commit();
	std::unique_lock<std::shared_mutex> wk(mu);
	auto& slot = m[key];
	if (slot.fn != nullptr)
	{
		// Lost the race: another thread inserted first. Drop our buffer.
		return slot.fn;
	}
	slot.buf = std::move(buf);
	slot.fn = fn;
	return fn;
}

nnc_gemv_f32_fn jit_kernel_cache::get_gemv_f32(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	const uint64_t key = pack(rows, cols);
	return reinterpret_cast<nnc_gemv_f32_fn>(
		lookup_or_build(p_->mu_, p_->gemv_, key,
		                [&](jit_buffer& b) { nnc_build_gemv_f32(b, rows, cols); }));
}

size_t jit_kernel_cache::size() const
{
	std::shared_lock<std::shared_mutex> lk(p_->mu_);
	return p_->gemv_.size() + p_->dot_f16_.size() + p_->gemv_f16w_f32x_.size()
		+ p_->gemv_bf16w_f32x_.size() + p_->gemv_q8_0_1row_.size();
}

nnc_dot_f16_fn jit_kernel_cache::get_dot_f16(const uint32_t n)
{
	assert(n > 0 && (n % 32) == 0);
	return reinterpret_cast<nnc_dot_f16_fn>(
		lookup_or_build(p_->mu_, p_->dot_f16_, n,
		                [&](jit_buffer& b) { nnc_build_dot_f16_to_f32(b, n); }));
}

nnc_gemv_f16w_f32x_fn jit_kernel_cache::get_gemv_f16w_f32x(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	const uint64_t key = pack(rows, cols);
	return reinterpret_cast<nnc_gemv_f16w_f32x_fn>(
		lookup_or_build(p_->mu_, p_->gemv_f16w_f32x_, key,
		                [&](jit_buffer& b) { nnc_build_gemv_f16w_f32x(b, rows, cols); }));
}

nnc_gemv_bf16w_f32x_fn jit_kernel_cache::get_gemv_bf16w_f32x(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	const uint64_t key = pack(rows, cols);
	return reinterpret_cast<nnc_gemv_bf16w_f32x_fn>(
		lookup_or_build(p_->mu_, p_->gemv_bf16w_f32x_, key,
		                [&](jit_buffer& b)
		                {
			                // Prefer the 4-rows-at-a-time builder when
			                // rows is a multiple of 4: it reuses each
			                // x-tile across 4 row accumulators, cutting
			                // x-side bandwidth by ~4x.
			                if ((rows % 4) == 0)
				                nnc_build_gemv_bf16w_f32x_4row(b, rows, cols);
			                else
				                nnc_build_gemv_bf16w_f32x(b, rows, cols);
		                }));
}

nnc_gemv_q8_0_1row_fn jit_kernel_cache::get_gemv_q8_0_1row(const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);
	return reinterpret_cast<nnc_gemv_q8_0_1row_fn>(
		lookup_or_build(p_->mu_, p_->gemv_q8_0_1row_, cols,
		                [&](jit_buffer& b) { nnc_build_gemv_q8_0_f32x_1row(b, cols); }));
}
