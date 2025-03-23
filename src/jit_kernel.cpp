// nnc — CPU feature detection and the JIT kernel cache.
// Detects AVX2/FMA/F16C via cpuid at startup, refuses to run on older
// CPUs, and owns the (op, dtype, shape)-keyed cache that JITs each kernel
// once and hands out reusable typed function pointers.

#include "jit_kernel.h"

#include "jit_buffer.h"
#include "jit_ops.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <intrin.h>
#include <unordered_map>

// ---- CPU detection -----------------------------------------------------

static cpu_features detect()
{
	cpu_features f{};

	int regs[4] = {0, 0, 0, 0};

	__cpuid(regs, 1);
	const bool osxsave = (regs[2] & (1 << 27)) != 0;
	const bool avx = (regs[2] & (1 << 28)) != 0;
	f.fma = (regs[2] & (1 << 12)) != 0;
	f.f16c = (regs[2] & (1 << 29)) != 0;

	__cpuidex(regs, 7, 0);
	const bool avx2_bit = (regs[1] & (1 << 5)) != 0;

	bool ymm_enabled = false;
	if (osxsave && avx)
	{
		const unsigned long long xcr0 = _xgetbv(0);
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
};

jit_kernel_cache::jit_kernel_cache() : p_(std::make_unique<impl>())
{
}

jit_kernel_cache::~jit_kernel_cache() = default;

static inline uint64_t pack(const uint32_t a, const uint32_t b)
{
	return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

nnc_gemv_f32_fn jit_kernel_cache::get_gemv_f32(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);

	const uint64_t key = pack(rows, cols);
	const auto it = p_->gemv_.find(key);
	if (it != p_->gemv_.end())
	{
		return reinterpret_cast<nnc_gemv_f32_fn>(it->second.fn);
	}

	auto buf = std::make_unique<jit_buffer>();
	nnc_build_gemv_f32(*buf, rows, cols);
	void* fn = buf->commit();

	auto& slot = p_->gemv_[key];
	slot.buf = std::move(buf);
	slot.fn = fn;
	return reinterpret_cast<nnc_gemv_f32_fn>(fn);
}

size_t jit_kernel_cache::size() const
{
	return p_->gemv_.size() + p_->dot_f16_.size() + p_->gemv_f16w_f32x_.size();
}

nnc_dot_f16_fn jit_kernel_cache::get_dot_f16(const uint32_t n)
{
	assert(n > 0 && (n % 32) == 0);

	const auto it = p_->dot_f16_.find(n);
	if (it != p_->dot_f16_.end())
	{
		return reinterpret_cast<nnc_dot_f16_fn>(it->second.fn);
	}

	auto buf = std::make_unique<jit_buffer>();
	nnc_build_dot_f16_to_f32(*buf, n);
	void* fn = buf->commit();

	auto& slot = p_->dot_f16_[n];
	slot.buf = std::move(buf);
	slot.fn = fn;
	return reinterpret_cast<nnc_dot_f16_fn>(fn);
}

nnc_gemv_f16w_f32x_fn jit_kernel_cache::get_gemv_f16w_f32x(const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);

	const uint64_t key = pack(rows, cols);
	const auto it = p_->gemv_f16w_f32x_.find(key);
	if (it != p_->gemv_f16w_f32x_.end())
	{
		return reinterpret_cast<nnc_gemv_f16w_f32x_fn>(it->second.fn);
	}

	auto buf = std::make_unique<jit_buffer>();
	nnc_build_gemv_f16w_f32x(*buf, rows, cols);
	void* fn = buf->commit();

	auto& slot = p_->gemv_f16w_f32x_[key];
	slot.buf = std::move(buf);
	slot.fn = fn;
	return reinterpret_cast<nnc_gemv_f16w_f32x_fn>(fn);
}
