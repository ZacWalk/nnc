// nnc — typed function-pointer wrappers, CPU detection, and the kernel cache.

#pragma once

#include <cstdint>
#include <memory>

class jit_buffer;

struct cpu_features
{
	bool avx2 = false;
	bool fma = false;
	bool f16c = false;
};

const cpu_features& nnc_cpu_features();
void nnc_require_avx2_fma();

// ---- Kernel signatures -------------------------------------------------

using nnc_dot_f32_fn = float (*)(const float* a, const float* b, size_t n);

// Specialized gemv: y[r] = sum_k W[r*cols + k] * x[k]   for r in [0, rows).
// W and x are FP32, contiguous, row-major. cols and rows are baked into
// the kernel — there are NO size args at runtime.
using nnc_gemv_f32_fn = void (*)(const float* W, const float* x, float* y);

// Specialized FP16 dot product:  return sum_i x[i] * y[i]  (i in [0, n)).
// n is baked into the kernel; n must be > 0 and a multiple of 32.
using nnc_dot_f16_fn = float (*)(const void* x, const void* y);

// Specialized fused gemv:
//   y[r] = sum_k fp16_to_fp32(W[r*cols + k]) * x[k]   for r in [0, rows).
// W is FP16, x and y are FP32. rows and cols are baked.
using nnc_gemv_f16w_f32x_fn = void (*)(const void* W, const float* x, float* y);

// ---- Kernel cache ------------------------------------------------------

class jit_kernel_cache
{
public:
	jit_kernel_cache();
	~jit_kernel_cache();

	jit_kernel_cache(const jit_kernel_cache&) = delete;
	jit_kernel_cache& operator=(const jit_kernel_cache&) = delete;

	// Returns a JITted gemv kernel for the given (rows, cols). cols must
	// be a positive multiple of 8. The kernel is built on first request
	// and reused thereafter. Returned pointer outlives the cache.
	nnc_gemv_f32_fn get_gemv_f32(uint32_t rows, uint32_t cols);

	// Returns a JITted FP16 dot kernel for the given n. n must be > 0 and
	// a multiple of 32.
	nnc_dot_f16_fn get_dot_f16(uint32_t n);

	// Returns a JITted fused FP16 W * FP32 x -> FP32 y gemv kernel for the
	// given (rows, cols). cols must be a positive multiple of 8.
	nnc_gemv_f16w_f32x_fn get_gemv_f16w_f32x(uint32_t rows, uint32_t cols);

	// For tests / introspection.
	size_t size() const;

private:
	struct impl;
	std::unique_ptr<impl> p_;
};
