// nnc — neural-net op implementations.
// Hand-written AVX2/FMA/F16C SIMD kernels (gelu, softmax, layernorm,
// elementwise add) plus thin wrappers that route dot/gemv to JITted
// kernels from jit_kernel.cpp. Also hosts the small graph-level fuser
// that collapses mul_mat -> repeat(bias) -> add [-> gelu] chains.

#include "nn_ops.h"

#include "jit_kernel.h"
#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

// Horizontal reduce: sum of 8 floats in a ymm register.
static inline float hsum_ymm(const __m256 v)
{
	const __m128 hi = _mm256_extractf128_ps(v, 1);
	const __m128 lo = _mm256_castps256_ps128(v);
	__m128 s = _mm_add_ps(lo, hi);
	s = _mm_hadd_ps(s, s);
	s = _mm_hadd_ps(s, s);
	return _mm_cvtss_f32(s);
}

// Horizontal reduce: max of 8 floats in a ymm register.
static inline float hmax_ymm(const __m256 v)
{
	const __m128 hi = _mm256_extractf128_ps(v, 1);
	const __m128 lo = _mm256_castps256_ps128(v);
	__m128 m = _mm_max_ps(lo, hi);
	m = _mm_max_ps(m, _mm_movehl_ps(m, m));
	m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
	return _mm_cvtss_f32(m);
}

// AVX2 exp(x) approximation. Uses range reduction x = n*ln(2) + r with r
// in [-ln(2)/2, ln(2)/2], then a 5-degree minimax polynomial for exp(r),
// followed by ldexp via integer add into the FP32 exponent. Max relative
// error ~3e-7 across [-87.3, 88.7]; outside that range the result
// saturates to 0 / +inf which is exactly what softmax wants for the
// negative-large / positive-large extremes (the negative side flushes
// underflows to 0; the positive side cannot occur because softmax has
// already subtracted the max).
static inline __m256 nnc_exp_ps(__m256 x)
{
	const __m256 v_lo = _mm256_set1_ps(-87.336544f); // ~ln(FLT_MIN)
	const __m256 v_hi = _mm256_set1_ps(88.722839f); // ~ln(FLT_MAX)
	x = _mm256_min_ps(_mm256_max_ps(x, v_lo), v_hi);

	const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
	const __m256 C1 = _mm256_set1_ps(0.693359375f); // ln(2) hi
	const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f); // ln(2) lo
	const __m256 ONE = _mm256_set1_ps(1.0f);

	// fx = round(x * log2(e))
	__m256 fx = _mm256_mul_ps(x, LOG2EF);
	fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	// r = x - fx*ln(2)  (computed in two halves for accuracy)
	__m256 r = _mm256_fnmadd_ps(fx, C1, x);
	r = _mm256_fnmadd_ps(fx, C2, r);

	// 5-degree polynomial for exp(r) on [-ln2/2, ln2/2].
	const __m256 P0 = _mm256_set1_ps(1.9875691500e-4f);
	const __m256 P1 = _mm256_set1_ps(1.3981999507e-3f);
	const __m256 P2 = _mm256_set1_ps(8.3334519073e-3f);
	const __m256 P3 = _mm256_set1_ps(4.1665795894e-2f);
	const __m256 P4 = _mm256_set1_ps(1.6666665459e-1f);
	const __m256 P5 = _mm256_set1_ps(5.0000001201e-1f);

	__m256 y = _mm256_fmadd_ps(P0, r, P1);
	y = _mm256_fmadd_ps(y, r, P2);
	y = _mm256_fmadd_ps(y, r, P3);
	y = _mm256_fmadd_ps(y, r, P4);
	y = _mm256_fmadd_ps(y, r, P5);
	const __m256 r2 = _mm256_mul_ps(r, r);
	y = _mm256_fmadd_ps(y, r2, r);
	y = _mm256_add_ps(y, ONE);

	// ldexp: result *= 2^fx
	const __m256i emm = _mm256_slli_epi32(
		_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
	return _mm256_mul_ps(y, _mm256_castsi256_ps(emm));
}

void nnc_gelu_f32(float* y, const float* x, const size_t n)
{
	// GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
	//
	// SIMD tanh: Padé[7/6] over [-5, 5], saturates to sign(u) for |u| >= 5.
	//   tanh(x) ~= x*(135135 + 17325*x^2 + 378*x^4 + x^6)
	//             / (135135 + 62370*x^2 + 3150*x^4 + 28*x^6)
	// Accuracy: < 5e-4 absolute in GELU output (tested).
	const __m256 v_a = _mm256_set1_ps(0.044715f);
	const __m256 v_c = _mm256_set1_ps(0.79788456080286535f); // sqrt(2/pi)
	const __m256 v_half = _mm256_set1_ps(0.5f);
	const __m256 v_one = _mm256_set1_ps(1.0f);
	const __m256 v_135135 = _mm256_set1_ps(135135.0f);
	const __m256 v_17325 = _mm256_set1_ps(17325.0f);
	const __m256 v_378 = _mm256_set1_ps(378.0f);
	const __m256 v_62370 = _mm256_set1_ps(62370.0f);
	const __m256 v_3150 = _mm256_set1_ps(3150.0f);
	const __m256 v_28 = _mm256_set1_ps(28.0f);
	const __m256 v_5 = _mm256_set1_ps(5.0f);
	const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);

	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		const __m256 v = _mm256_loadu_ps(x + i);
		const __m256 v2 = _mm256_mul_ps(v, v);
		const __m256 v3 = _mm256_mul_ps(v2, v);
		const __m256 u = _mm256_mul_ps(v_c, _mm256_fmadd_ps(v_a, v3, v));

		const __m256 u2 = _mm256_mul_ps(u, u);
		const __m256 u4 = _mm256_mul_ps(u2, u2);
		const __m256 u6 = _mm256_mul_ps(u4, u2);

		// num = u*(135135 + 17325*u^2 + 378*u^4 + u^6)
		__m256 num = _mm256_add_ps(u6,
		                           _mm256_fmadd_ps(v_378, u4,
		                                           _mm256_fmadd_ps(v_17325, u2, v_135135)));
		num = _mm256_mul_ps(u, num);
		// den = 135135 + 62370*u^2 + 3150*u^4 + 28*u^6
		const __m256 den = _mm256_fmadd_ps(v_28, u6,
		                                   _mm256_fmadd_ps(v_3150, u4,
		                                                   _mm256_fmadd_ps(v_62370, u2, v_135135)));
		const __m256 t_pade = _mm256_div_ps(num, den);

		const __m256 u_abs = _mm256_andnot_ps(v_sign_mask, u);
		const __m256 sat = _mm256_cmp_ps(u_abs, v_5, _CMP_GE_OQ);
		const __m256 sign_u = _mm256_or_ps(_mm256_and_ps(u, v_sign_mask), v_one);

		const __m256 t = _mm256_blendv_ps(t_pade, sign_u, sat);
		const __m256 r = _mm256_mul_ps(_mm256_mul_ps(v_half, v),
		                               _mm256_add_ps(v_one, t));
		_mm256_storeu_ps(y + i, r);
	}

	constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
	constexpr float GELU_COEF_A = 0.044715f;
	for (; i < n; ++i)
	{
		const float v = x[i];
		const float u = SQRT_2_OVER_PI * (v + GELU_COEF_A * v * v * v);
		y[i] = 0.5f * v * (1.0f + std::tanh(u));
	}
}

// ---- FP16 dot product --------------------------------------------------

// IEEE 754 binary16 -> binary32, software fallback (no F16C dependency).
// Used only for the scalar leftover path; the JIT path uses VCVTPH2PS.
static inline float fp16_to_fp32_scalar(const uint16_t h)
{
	const uint32_t s = (h >> 15) & 0x1u;
	const uint32_t e = (h >> 10) & 0x1Fu;
	const uint32_t m = h & 0x3FFu;

	uint32_t bits;
	if (e == 0)
	{
		if (m == 0)
		{
			bits = s << 31;
		}
		else
		{
			// subnormal: normalize. mm initially equals m; shift left until
			// bit 10 (0x400) is set — that bit becomes the implicit leading 1.
			// Number of shifts performed = 10 - (position of MSB in m).
			// The final exponent of the half-precision value is
			//   -14 - shifts (since min normal half = 2^-14, and each shift
			//   halves the magnitude).
			// Biased single-precision exponent = (-14 - shifts) + 127.
			uint32_t mm = m;
			uint32_t shifts = 0;
			while ((mm & 0x400u) == 0)
			{
				mm <<= 1;
				++shifts;
			}
			mm &= 0x3FFu;
			bits = (s << 31)
				| (static_cast<uint32_t>(127 - 14 - shifts) << 23)
				| (mm << 13);
		}
	}
	else if (e == 31)
	{
		bits = (s << 31) | (0xFFu << 23) | (m << 13);
	}
	else
	{
		bits = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
	}

	float f;
	std::memcpy(&f, &bits, 4);
	return f;
}

static float dot_f16_scalar(const uint16_t* x, const uint16_t* y, const size_t n)
{
	double s = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		s += static_cast<double>(fp16_to_fp32_scalar(x[i]))
			* static_cast<double>(fp16_to_fp32_scalar(y[i]));
	}
	return static_cast<float>(s);
}

namespace
{
	jit_kernel_cache& global_cache()
	{
		static jit_kernel_cache c;
		return c;
	}
}

float nnc_dot_f16_to_f32(const void* x, const void* y, const size_t n)
{
	if (n == 0) return 0.0f;

	if ((n & 31u) == 0 && n <= UINT32_MAX)
	{
		const auto fn = global_cache().get_dot_f16(static_cast<uint32_t>(n));
		return fn(x, y);
	}

	return dot_f16_scalar(static_cast<const uint16_t*>(x),
	                      static_cast<const uint16_t*>(y), n);
}

// ---- softmax (in place) ------------------------------------------------

void nnc_softmax_f32_inplace(float* p, const size_t n)
{
	if (n == 0) return;

	// Pass 1: find max via AVX2.
	float m = -INFINITY;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vmax = _mm256_loadu_ps(p);
			i = 8;
			for (; i + 8 <= n; i += 8)
			{
				vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(p + i));
			}
			m = hmax_ymm(vmax);
		}
		for (; i < n; ++i)
		{
			if (p[i] > m) m = p[i];
		}
	}

	// Pass 2: exp(p[i] - m), accumulate sum. AVX2 polynomial nnc_exp_ps
	// keeps softmax fully vectorised. -INFINITY entries become 0 because
	// nnc_exp_ps clamps to ~ln(FLT_MIN) and exp() of that underflows.
	double sum = 0.0;
	{
		const __m256 vm = _mm256_set1_ps(m);
		__m256 vsum = _mm256_setzero_ps();
		size_t i = 0;
		for (; i + 8 <= n; i += 8)
		{
			__m256 v = _mm256_loadu_ps(p + i);
			v = nnc_exp_ps(_mm256_sub_ps(v, vm));
			_mm256_storeu_ps(p + i, v);
			vsum = _mm256_add_ps(vsum, v);
		}
		sum = static_cast<double>(hsum_ymm(vsum));
		for (; i < n; ++i)
		{
			if (p[i] == -INFINITY)
			{
				p[i] = 0.0f;
			}
			else
			{
				const float v = std::exp(p[i] - m);
				p[i] = v;
				sum += v;
			}
		}
	}

	// Pass 3: scale by 1/sum via AVX2. If `sum` is zero (e.g. every input
	// was -INFINITY because the entire row is masked), the outputs are
	// already zero from pass 2 — guard against producing NaN.
	const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
	const __m256 vinv = _mm256_set1_ps(inv);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
	}
	for (; i < n; ++i) p[i] *= inv;
}

// ---- fused per-head attention softmax + V matmul -----------------------

void nnc_attn_softmax_v_f32(float* out, float* scores, const float* V,
                            const size_t n_t, const size_t v_stride, const size_t head_dim)
{
	if (n_t == 0 || head_dim == 0) return;

	// Reuse the existing fully-vectorised softmax (max + AVX2 exp + scale).
	nnc_softmax_f32_inplace(scores, n_t);

	// Weighted sum of V rows, written to `out` in a single sweep. The
	// "first row writes, rest accumulate" trick removes the explicit
	// memset(out) the unfused path required.
	for (size_t t = 0; t < n_t; ++t)
	{
		const float w = scores[t];
		const float* vrow = V + t * v_stride;
		const __m256 vw = _mm256_set1_ps(w);
		size_t i = 0;
		if (t == 0)
		{
			for (; i + 8 <= head_dim; i += 8)
				_mm256_storeu_ps(out + i,
				                 _mm256_mul_ps(vw, _mm256_loadu_ps(vrow + i)));
			for (; i < head_dim; ++i) out[i] = w * vrow[i];
		}
		else
		{
			for (; i + 8 <= head_dim; i += 8)
			{
				const __m256 acc = _mm256_loadu_ps(out + i);
				const __m256 v = _mm256_loadu_ps(vrow + i);
				_mm256_storeu_ps(out + i, _mm256_fmadd_ps(vw, v, acc));
			}
			for (; i < head_dim; ++i) out[i] += w * vrow[i];
		}
	}
}

// ---- layernorm (no affine; matches NNC_OP_NORM) ------------------

void nnc_layernorm_f32(float* y, const float* x, const size_t n, const float eps)
{
	if (n == 0) return;

	// Pass 1: sum.
	double mean = 0.0;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vs = _mm256_setzero_ps();
			for (; i + 8 <= n; i += 8)
			{
				vs = _mm256_add_ps(vs, _mm256_loadu_ps(x + i));
			}
			mean = static_cast<double>(hsum_ymm(vs));
		}
		for (; i < n; ++i) mean += x[i];
		mean /= static_cast<double>(n);
	}

	// Pass 2: y[i] = x[i] - mean ; accumulate sum-of-squares.
	double sum2 = 0.0;
	{
		const __m256 vmean = _mm256_set1_ps(static_cast<float>(mean));
		__m256 vs2 = _mm256_setzero_ps();
		size_t i = 0;
		for (; i + 8 <= n; i += 8)
		{
			const __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
			_mm256_storeu_ps(y + i, v);
			vs2 = _mm256_fmadd_ps(v, v, vs2);
		}
		sum2 = static_cast<double>(hsum_ymm(vs2));
		for (; i < n; ++i)
		{
			const float v = static_cast<float>(x[i] - mean);
			y[i] = v;
			sum2 += static_cast<double>(v) * v;
		}
	}

	// Pass 3: y[i] *= scale.
	const float scale = static_cast<float>(1.0 / std::sqrt(sum2 / static_cast<double>(n) + eps));
	const __m256 vscale = _mm256_set1_ps(scale);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vscale));
	}
	for (; i < n; ++i) y[i] *= scale;
}

// ---- rmsnorm (no affine; y[i] = x[i] / sqrt(mean(x^2) + eps)) ----

void nnc_rmsnorm_f32(float* y, const float* x, const size_t n, const float eps)
{
	if (n == 0) return;

	// Pass 1: sum-of-squares.
	double sum2 = 0.0;
	{
		size_t i = 0;
		if (n >= 8)
		{
			__m256 vs2 = _mm256_setzero_ps();
			for (; i + 8 <= n; i += 8)
			{
				const __m256 v = _mm256_loadu_ps(x + i);
				vs2 = _mm256_fmadd_ps(v, v, vs2);
			}
			sum2 = static_cast<double>(hsum_ymm(vs2));
		}
		for (; i < n; ++i) sum2 += static_cast<double>(x[i]) * x[i];
	}

	// Pass 2: y[i] = x[i] * scale.
	const float scale = static_cast<float>(1.0 / std::sqrt(sum2 / static_cast<double>(n) + eps));
	const __m256 vscale = _mm256_set1_ps(scale);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vscale));
	}
	for (; i < n; ++i) y[i] = x[i] * scale;
}

void nnc_rmsnorm_gamma_multi_f32(float* y, const float* x,
                                 const size_t n_groups, const size_t dim,
                                 const float* gamma, const float eps)
{
	if (n_groups == 0 || dim == 0) return;
	const float inv_dim = 1.0f / static_cast<float>(dim);
	for (size_t g = 0; g < n_groups; ++g)
	{
		const float* xg = x + g * dim;
		float* yg = y + g * dim;

		// Pass 1: sum-of-squares.
		double sum2 = 0.0;
		{
			size_t i = 0;
			if (dim >= 8)
			{
				__m256 vs2 = _mm256_setzero_ps();
				for (; i + 8 <= dim; i += 8)
				{
					const __m256 v = _mm256_loadu_ps(xg + i);
					vs2 = _mm256_fmadd_ps(v, v, vs2);
				}
				sum2 = static_cast<double>(hsum_ymm(vs2));
			}
			for (; i < dim; ++i) sum2 += static_cast<double>(xg[i]) * xg[i];
		}
		const float scale = static_cast<float>(
			1.0 / std::sqrt(sum2 * inv_dim + eps));

		// Pass 2 fused with gamma multiply: y[i] = x[i] * scale * gamma[i].
		const __m256 vscale = _mm256_set1_ps(scale);
		size_t i = 0;
		for (; i + 8 <= dim; i += 8)
		{
			__m256 v = _mm256_loadu_ps(xg + i);
			v = _mm256_mul_ps(v, vscale);
			v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + i));
			_mm256_storeu_ps(yg + i, v);
		}
		for (; i < dim; ++i) yg[i] = xg[i] * scale * gamma[i];
	}
}

float nnc_dot_f32_simd(const float* a, const float* b, const size_t n)
{
	size_t i = 0;
	__m256 s0 = _mm256_setzero_ps();
	__m256 s1 = _mm256_setzero_ps();
	__m256 s2 = _mm256_setzero_ps();
	__m256 s3 = _mm256_setzero_ps();
	for (; i + 32 <= n; i += 32)
	{
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0),
		                     _mm256_loadu_ps(b + i + 0), s0);
		s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
		                     _mm256_loadu_ps(b + i + 8), s1);
		s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
		                     _mm256_loadu_ps(b + i + 16), s2);
		s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
		                     _mm256_loadu_ps(b + i + 24), s3);
	}
	for (; i + 8 <= n; i += 8)
	{
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
		                     _mm256_loadu_ps(b + i), s0);
	}
	s0 = _mm256_add_ps(s0, s1);
	s2 = _mm256_add_ps(s2, s3);
	s0 = _mm256_add_ps(s0, s2);
	float acc = hsum_ymm(s0);
	for (; i < n; ++i) acc += a[i] * b[i];
	return acc;
}

// ---- fused FP16 W * FP32 x -> FP32 y gemv ------------------------------

void nnc_gemv_f16w_f32x(const void* W, const float* x, float* y,
                        const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;

	if ((cols & 7u) == 0)
	{
		const auto fn = global_cache().get_gemv_f16w_f32x(rows, cols);
		fn(W, x, y);
		return;
	}

	// Scalar fallback (cols not a multiple of 8). Uses the same software
	// FP16 decode as the dot fallback for bit-identical behaviour with the
	// hardware F16C path on normal inputs.
	const auto Wh = static_cast<const uint16_t*>(W);
	for (uint32_t r = 0; r < rows; ++r)
	{
		double s = 0.0;
		const uint16_t* row = Wh + static_cast<size_t>(r) * cols;
		for (uint32_t k = 0; k < cols; ++k)
		{
			s += static_cast<double>(fp16_to_fp32_scalar(row[k])) * x[k];
		}
		y[r] = static_cast<float>(s);
	}
}

// ---- elementwise bias add ----------------------------------------------

void nnc_add_inplace_f32(float* y, const float* b, const size_t n)
{
	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(b + i)));
	}
	for (; i < n; ++i) y[i] += b[i];
}

// ---- elementwise multiply ----------------------------------------------

void nnc_mul_inplace_f32(float* y, const float* s, const size_t n)
{
	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(s + i)));
	}
	for (; i < n; ++i) y[i] *= s[i];
}

// ---- fused GELU * up (gated-MLP inner step) ----------------------------
//
// y[i] = gelu(gate[i]) * up[i]. Reuses the AVX2 Pade[7/6] tanh path from
// nnc_gelu_f32 — we just multiply the GELU result by up[i] before the
// store, eliminating one full read+write pass over the n_ff activation
// tile (~32KB / layer for Gemma E2B at n_ff=8192).
void nnc_gelu_mul_f32(float* y, const float* gate, const float* up, const size_t n)
{
	const __m256 v_a = _mm256_set1_ps(0.044715f);
	const __m256 v_c = _mm256_set1_ps(0.79788456080286535f);
	const __m256 v_half = _mm256_set1_ps(0.5f);
	const __m256 v_one = _mm256_set1_ps(1.0f);
	const __m256 v_135135 = _mm256_set1_ps(135135.0f);
	const __m256 v_17325 = _mm256_set1_ps(17325.0f);
	const __m256 v_378 = _mm256_set1_ps(378.0f);
	const __m256 v_62370 = _mm256_set1_ps(62370.0f);
	const __m256 v_3150 = _mm256_set1_ps(3150.0f);
	const __m256 v_28 = _mm256_set1_ps(28.0f);
	const __m256 v_5 = _mm256_set1_ps(5.0f);
	const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);

	size_t i = 0;
	const size_t np = n & ~size_t{7};
	for (; i < np; i += 8)
	{
		const __m256 v = _mm256_loadu_ps(gate + i);
		const __m256 u_hi = _mm256_loadu_ps(up + i);
		const __m256 v2 = _mm256_mul_ps(v, v);
		const __m256 v3 = _mm256_mul_ps(v2, v);
		const __m256 u = _mm256_mul_ps(v_c, _mm256_fmadd_ps(v_a, v3, v));
		const __m256 u2 = _mm256_mul_ps(u, u);
		const __m256 u4 = _mm256_mul_ps(u2, u2);
		const __m256 u6 = _mm256_mul_ps(u4, u2);

		__m256 num = _mm256_add_ps(u6,
		                           _mm256_fmadd_ps(v_378, u4,
		                                           _mm256_fmadd_ps(v_17325, u2, v_135135)));
		num = _mm256_mul_ps(u, num);
		const __m256 den = _mm256_fmadd_ps(v_28, u6,
		                                   _mm256_fmadd_ps(v_3150, u4,
		                                                   _mm256_fmadd_ps(v_62370, u2, v_135135)));
		const __m256 t_pade = _mm256_div_ps(num, den);

		const __m256 u_abs = _mm256_andnot_ps(v_sign_mask, u);
		const __m256 sat = _mm256_cmp_ps(u_abs, v_5, _CMP_GE_OQ);
		const __m256 sign_u = _mm256_or_ps(_mm256_and_ps(u, v_sign_mask), v_one);
		const __m256 t = _mm256_blendv_ps(t_pade, sign_u, sat);
		__m256 r = _mm256_mul_ps(_mm256_mul_ps(v_half, v),
		                         _mm256_add_ps(v_one, t));
		r = _mm256_mul_ps(r, u_hi);
		_mm256_storeu_ps(y + i, r);
	}

	constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
	constexpr float GELU_COEF_A = 0.044715f;
	for (; i < n; ++i)
	{
		const float v = gate[i];
		const float uu = SQRT_2_OVER_PI * (v + GELU_COEF_A * v * v * v);
		y[i] = 0.5f * v * (1.0f + std::tanh(uu)) * up[i];
	}
}

// ---- BF16-weight, FP32-activation gemv ---------------------------------
//
// AVX2 path processes 8 cols at a time. BF16 -> F32 is a free shift:
//   f32 = (u32)(bf16) << 16
// Implemented as vpmovzxwd (zero-extend 8 u16 to 8 u32) + vpslld(16) +
// vfmadd231ps. Falls through to a scalar reference for the column tail
// or when cols < 8.
//
// Multi-threaded fast path: when the gemv is "big enough" (rows >= 256 and
// cols >= 256) and rows is a multiple of 4, we split the row axis across
// a static worker pool. Each worker calls a single shared (rows=4)-baked
// JIT kernel on its slice — this keeps the kernel cache to one entry per
// `cols` while still benefitting from the 4-row inner-loop. The pool is
// created lazily on first use (workers spin briefly then yield while
// idle, avoiding condition-variable wakeup latency on the hot path).

namespace
{
	// Lightweight bring-your-own-task pool. Workers spin a few thousand
	// pause/yield iterations on `cur_ticket` advancing, then run
	// `task_fn(tid+1, task_user)`. Dispatch latency is sub-microsecond
	// when workers are warm, far below the cost of any gemv we care
	// about parallelising.
	class nnc_gemv_pool
	{
	public:
		using task_fn_t = void (*)(const int tid, const void* user);

		static nnc_gemv_pool& global()
		{
			static nnc_gemv_pool p(decide_n_workers());
			return p;
		}

		// Total participating threads = main + workers.
		int n_threads() const { return static_cast<int>(workers_.size()) + 1; }

		// Run `fn(tid, user)` for tid in [0, n_threads()). Main thread
		// runs tid=0; the n_workers workers run tids 1..n. Blocks until
		// all participants finish.
		void dispatch(const task_fn_t fn, void* user)
		{
			const int nw = static_cast<int>(workers_.size());
			task_fn_ = fn;
			task_user_ = user;
			// Snapshot per-worker baselines, advance ticket, run main
			// share, spin-wait for workers.
			unsigned baselines[64];
			for (int i = 0; i < nw; ++i)
				baselines[i] = workers_[i]->done.load(std::memory_order_acquire);
			cur_ticket_.fetch_add(1, std::memory_order_release);
			fn(0, user);
			for (int i = 0; i < nw; ++i)
			{
				while (workers_[i]->done.load(std::memory_order_acquire) == baselines[i])
					_mm_pause();
			}
		}

		~nnc_gemv_pool()
		{
			stop_.store(true, std::memory_order_release);
			cur_ticket_.fetch_add(1, std::memory_order_release);
			for (const auto& w : workers_) w->thr.join();
		}

	private:
		struct slot
		{
			std::thread thr;
			std::atomic<unsigned> done{0};
		};

		static int decide_n_workers()
		{
			// Env override for perf experiments: NNC_THREADS = total
			// threads (workers + main). Otherwise default cap at 8.
			if (const char* e = std::getenv("NNC_THREADS"))
			{
				const int t = std::atoi(e);
				if (t > 0) return (t > 1 ? t - 1 : 0);
			}
			const unsigned hw = std::thread::hardware_concurrency();
			const unsigned t = (hw > 0 ? hw : 1);
			const unsigned use = (t > 8) ? 8u : t;
			return (use > 1 ? static_cast<int>(use) - 1 : 0);
		}

		explicit nnc_gemv_pool(const int n)
		{
			workers_.reserve(n);
			for (int i = 0; i < n; ++i)
			{
				workers_.push_back(std::make_unique<slot>());
				workers_.back()->thr = std::thread([this, i] { worker_loop(i); });
			}
		}

		void worker_loop(const int i)
		{
			unsigned last = 0;
			auto& w = *workers_[i];
			while (!stop_.load(std::memory_order_acquire))
			{
				unsigned t;
				int spins = 0;
				for (;;)
				{
					t = cur_ticket_.load(std::memory_order_acquire);
					if (t != last) break;
					if (stop_.load(std::memory_order_acquire)) return;
					if (++spins > 4096)
					{
						std::this_thread::yield();
						spins = 0;
					}
					else
					{
						_mm_pause();
					}
				}
				last = t;
				if (stop_.load(std::memory_order_acquire)) return;
				task_fn_(i + 1, task_user_);
				w.done.fetch_add(1, std::memory_order_release);
			}
		}

		std::vector<std::unique_ptr<slot>> workers_;
		std::atomic<unsigned> cur_ticket_{0};
		std::atomic<bool> stop_{false};
		task_fn_t task_fn_ = nullptr;
		void* task_user_ = nullptr;
	};

	struct bf16_gemv_ctx
	{
		const uint8_t* W;
		const float* x;
		float* y;
		uint32_t cols;
		uint32_t total_quads; // rows / 4
		int n_threads;
		nnc_gemv_bf16w_f32x_fn fn4;
	};

	void bf16_gemv_worker(const int tid, const void* u)
	{
		const auto* c = static_cast<const bf16_gemv_ctx*>(u);
		const uint32_t per = c->total_quads / static_cast<uint32_t>(c->n_threads);
		const uint32_t rem = c->total_quads % static_cast<uint32_t>(c->n_threads);
		const uint32_t q0 = static_cast<uint32_t>(tid) * per
			+ std::min(static_cast<uint32_t>(tid), rem);
		const uint32_t q1 = q0 + per + (static_cast<uint32_t>(tid) < rem ? 1u : 0u);
		if (q0 == q1) return;
		const size_t row_stride = static_cast<size_t>(c->cols) * 2;
		const size_t group_stride = row_stride * 4;
		const uint8_t* W_tid = c->W + q0 * group_stride;
		float* y_tid = c->y + q0 * 4;
		for (uint32_t q = q0; q < q1; ++q)
		{
			c->fn4(W_tid, c->x, y_tid);
			W_tid += group_stride;
			y_tid += 4;
		}
	}
}

void nnc_gemv_bf16w_f32x(const void* W, const float* x, float* y,
                         const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;

	if ((cols & 7u) == 0)
	{
		// Parallel path: rows must be a multiple of 4 (so we can use the
		// shared 4-row JIT kernel) and the gemv must be big enough to
		// amortise dispatch overhead. Threshold ~256 rows captures every
		// significant Gemma weight (Q/K/V/O/gate/up/down + lm_head) but
		// keeps tiny kernels (e.g. PLE on small ple_dim) on the fast
		// single-threaded path.
		auto& pool = nnc_gemv_pool::global();
		const int n_threads = pool.n_threads();
		if (n_threads > 1 && (rows & 3u) == 0 && rows >= 256 && cols >= 256)
		{
			const auto fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
			bf16_gemv_ctx ctx{
				static_cast<const uint8_t*>(W), x, y, cols,
				rows / 4, n_threads, fn4
			};
			pool.dispatch(&bf16_gemv_worker, &ctx);
			return;
		}

		const auto fn = global_cache().get_gemv_bf16w_f32x(rows, cols);
		fn(W, x, y);
		return;
	}

	// Scalar fallback (cols not a multiple of 8).
	const auto Wb = static_cast<const uint16_t*>(W);
	for (uint32_t r = 0; r < rows; ++r)
	{
		const uint16_t* row = Wb + static_cast<size_t>(r) * cols;
		double s = 0.0;
		for (uint32_t k = 0; k < cols; ++k)
		{
			const uint32_t u = static_cast<uint32_t>(row[k]) << 16;
			float wf;
			std::memcpy(&wf, &u, 4);
			s += static_cast<double>(wf) * x[k];
		}
		y[r] = static_cast<float>(s);
	}
}

// ---- Streaming BF16-weight gemv + argmax -------------------------------
//
// For lm_head / final logits projection in greedy decode: compute the
// gemv 4 rows at a time directly into a tiny scratch and update a
// running argmax. Never materialises the full vocab-sized logits buffer.
// Saves ~1 MB of writes per token (vocab=262144) plus the softcap pass
// (softcap is monotonic and doesn't change argmax).

int nnc_gemv_bf16w_argmax_f32x(const void* W, const float* x,
                               const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return -1;

	if ((rows & 3u) == 0 && (cols & 7u) == 0)
	{
		const auto fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
		const auto Wb = static_cast<const uint8_t*>(W);
		const size_t row_stride = static_cast<size_t>(cols) * 2; // BF16 bytes/row

		// Parallel path: split the row axis. Each worker computes its
		// local (best_val, best_idx); main reduces across workers. The
		// argmax is the dominant cost at lm_head (rows = vocab = 262144),
		// so this is the highest-payoff parallel site in the model.
		auto& pool = nnc_gemv_pool::global();
		const int n_threads = pool.n_threads();
		if (n_threads > 1 && rows >= 256 && cols >= 256)
		{
			struct local
			{
				float bv;
				int bi;
			};
			alignas(64) local locals[64]{};
			for (int i = 0; i < n_threads; ++i)
			{
				locals[i].bv = -std::numeric_limits<float>::infinity();
				locals[i].bi = -1;
			}

			struct argmax_ctx
			{
				const uint8_t* W;
				const float* x;
				size_t row_stride;
				uint32_t total_quads;
				int n_threads;
				nnc_gemv_bf16w_f32x_fn fn4;
				local* locals;
			} c{
				Wb, x, row_stride, rows / 4, n_threads, fn4, locals
			};

			pool.dispatch([](const int tid, const void* u)
			{
				const auto* cc = static_cast<const argmax_ctx*>(u);
				const uint32_t per = cc->total_quads / static_cast<uint32_t>(cc->n_threads);
				const uint32_t rem = cc->total_quads % static_cast<uint32_t>(cc->n_threads);
				const uint32_t q0 = static_cast<uint32_t>(tid) * per
					+ std::min(static_cast<uint32_t>(tid), rem);
				const uint32_t q1 = q0 + per + (static_cast<uint32_t>(tid) < rem ? 1u : 0u);
				if (q0 == q1) return;
				alignas(16) float scratch[4];
				float bv = -std::numeric_limits<float>::infinity();
				int bi = -1;
				const size_t group_stride = cc->row_stride * 4;
				const uint8_t* Wp = cc->W + q0 * group_stride;
				for (uint32_t q = q0; q < q1; ++q)
				{
					cc->fn4(Wp, cc->x, scratch);
					const int base = static_cast<int>(q) * 4;
					if (scratch[0] > bv)
					{
						bv = scratch[0];
						bi = base;
					}
					if (scratch[1] > bv)
					{
						bv = scratch[1];
						bi = base + 1;
					}
					if (scratch[2] > bv)
					{
						bv = scratch[2];
						bi = base + 2;
					}
					if (scratch[3] > bv)
					{
						bv = scratch[3];
						bi = base + 3;
					}
					Wp += group_stride;
				}
				cc->locals[tid].bv = bv;
				cc->locals[tid].bi = bi;
			}, &c);

			int best = locals[0].bi;
			float bv = locals[0].bv;
			for (int i = 1; i < n_threads; ++i)
			{
				if (locals[i].bi >= 0 && locals[i].bv > bv)
				{
					bv = locals[i].bv;
					best = locals[i].bi;
				}
			}
			return best;
		}

		// Single-threaded fast path.
		alignas(16) float scratch[4];
		int best = 0;
		float bv = -std::numeric_limits<float>::infinity();
		for (uint32_t r = 0; r < rows; r += 4)
		{
			fn4(Wb + static_cast<size_t>(r) * row_stride, x, scratch);
			if (scratch[0] > bv)
			{
				bv = scratch[0];
				best = static_cast<int>(r);
			}
			if (scratch[1] > bv)
			{
				bv = scratch[1];
				best = static_cast<int>(r) + 1;
			}
			if (scratch[2] > bv)
			{
				bv = scratch[2];
				best = static_cast<int>(r) + 2;
			}
			if (scratch[3] > bv)
			{
				bv = scratch[3];
				best = static_cast<int>(r) + 3;
			}
		}
		return best;
	}

	// Generic fallback: full materialise then linear scan.
	std::vector<float> tmp(rows);
	nnc_gemv_bf16w_f32x(W, x, tmp.data(), rows, cols);
	int best = 0;
	float bv = tmp[0];
	for (uint32_t r = 1; r < rows; ++r)
	{
		if (tmp[r] > bv)
		{
			bv = tmp[r];
			best = static_cast<int>(r);
		}
	}
	return best;
}

// ---- Q8_0 quantize + gemv ---------------------------------------------
//
// Block size 32. For each row's block, scale = max(|w|) / 127 and
// q[k] = round(w[k] / scale) clamped to [-127, 127]. Standard Q8_0
// (matching what GGML / llama.cpp ships) but written into the SPLIT
// layout (qs first, scales second) so the JIT kernel can use cheap
// scaled-q + FMA without an interleaved per-block fp16 unpack.

void nnc_quantize_bf16_to_q8_0(const uint16_t* W_bf16, int8_t* qs, float* scales,
                               const size_t rows, const size_t cols)
{
	NNC_ASSERT(W_bf16 && qs && scales);
	NNC_ASSERT(cols > 0 && (cols % 32) == 0);

	const size_t nblocks = cols / 32;
	for (size_t r = 0; r < rows; ++r)
	{
		const uint16_t* row_w = W_bf16 + r * cols;
		int8_t* row_q = qs + r * cols;
		float* row_s = scales + r * nblocks;
		for (size_t b = 0; b < nblocks; ++b)
		{
			float amax = 0.0f;
			float vals[32];
			for (size_t k = 0; k < 32; ++k)
			{
				vals[k] = nnc_bf16_to_f32(row_w[b * 32 + k]);
				const float a = std::fabs(vals[k]);
				if (a > amax) amax = a;
			}
			const float scale = amax / 127.0f;
			row_s[b] = scale;
			const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
			for (size_t k = 0; k < 32; ++k)
			{
				int q = static_cast<int>(std::lrintf(vals[k] * inv));
				if (q > 127) q = 127;
				if (q < -127) q = -127;
				row_q[b * 32 + k] = static_cast<int8_t>(q);
			}
		}
	}
}

namespace
{
	struct q8_gemv_ctx
	{
		const int8_t* qs;
		const float* scales;
		const float* x;
		float* y;
		uint32_t cols;
		uint32_t rows;
		int n_threads;
		nnc_gemv_q8_0_1row_fn fn1;
	};

	void q8_gemv_worker(const int tid, const void* u)
	{
		const auto* c = static_cast<const q8_gemv_ctx*>(u);
		const uint32_t per = c->rows / static_cast<uint32_t>(c->n_threads);
		const uint32_t rem = c->rows % static_cast<uint32_t>(c->n_threads);
		const uint32_t r0 = static_cast<uint32_t>(tid) * per
			+ std::min(static_cast<uint32_t>(tid), rem);
		const uint32_t r1 = r0 + per + (static_cast<uint32_t>(tid) < rem ? 1u : 0u);
		if (r0 == r1) return;
		const size_t qs_stride = c->cols;
		const size_t scales_stride = c->cols / 32;
		const int8_t* qs_tid = c->qs + r0 * qs_stride;
		const float* scales_tid = c->scales + r0 * scales_stride;
		float* y_tid = c->y + r0;
		for (uint32_t r = r0; r < r1; ++r)
		{
			c->fn1(qs_tid, c->x, y_tid, scales_tid);
			qs_tid += qs_stride;
			scales_tid += scales_stride;
			y_tid += 1;
		}
	}
}

void nnc_gemv_q8_0_f32x(const int8_t* qs, const float* scales,
                        const float* x, float* y,
                        const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;
	NNC_ASSERT((cols % 32) == 0);

	const auto fn1 = global_cache().get_gemv_q8_0_1row(cols);

	auto& pool = nnc_gemv_pool::global();
	const int n_threads = pool.n_threads();
	if (n_threads > 1 && rows >= 256 && cols >= 256)
	{
		q8_gemv_ctx ctx{qs, scales, x, y, cols, rows, n_threads, fn1};
		pool.dispatch(&q8_gemv_worker, &ctx);
		return;
	}

	const size_t qs_stride = cols;
	const size_t scales_stride = cols / 32;
	for (uint32_t r = 0; r < rows; ++r)
	{
		fn1(qs + r * qs_stride, x, y + r, scales + r * scales_stride);
	}
}

// ---- SwiGLU ------------------------------------------------------------
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// y[i] = silu(gate[i]) * up[i]
//
// Uses the existing exp_ps approximation if available; otherwise the
// std::exp scalar path. AVX2 inner uses 1 / (1 + exp(-g)) per lane via
// rcp_ps or a Cephes-style poly. Keep it simple: scalar std::exp inside
// an 8-wide gather/scatter wrapper. This is not in the model's hottest
// inner loop so accuracy beats throughput here.

void nnc_swiglu_f32(float* y, const float* gate, const float* up, const size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		const float g = gate[i];
		const float sig = 1.0f / (1.0f + std::exp(-g));
		y[i] = g * sig * up[i];
	}
}

// ---- RoPE (NeoX / half-pair convention) --------------------------------

void nnc_rope_f32(float* x, const uint32_t n_heads, const uint32_t head_dim,
                  const uint32_t n_rot, const float pos, const float freq_base)
{
	if (n_rot == 0) return;
	NNC_ASSERT((n_rot & 1u) == 0 && "nnc_rope_f32: n_rot must be even");
	const uint32_t half = n_rot / 2;
	const float p = pos;
	// freq_base ^ (-2/n_rot) is the per-step ratio of the geometric inv
	// schedule. Compute once, then accumulate; avoids one std::pow per
	// (head, pair) on the hot path (~16x faster than the per-element
	// std::pow it replaces for n_rot=256).
	const float ratio = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
	for (uint32_t h = 0; h < n_heads; ++h)
	{
		float* xh = x + static_cast<size_t>(h) * head_dim;
		float inv = 1.0f;
		for (uint32_t i = 0; i < half; ++i)
		{
			const float theta = p * inv;
			const float c = std::cos(theta);
			const float s = std::sin(theta);
			const float a = xh[i];
			const float b = xh[i + half];
			xh[i] = c * a - s * b;
			xh[i + half] = s * a + c * b;
			inv *= ratio;
		}
	}
}

// ---- soft-cap (tanh(x/c)*c) --------------------------------------------

void nnc_softcap_f32(float* y, const float* x, const size_t n, const float cap)
{
	const float inv = 1.0f / cap;
	for (size_t i = 0; i < n; ++i)
		y[i] = std::tanh(x[i] * inv) * cap;
}

// ---- BF16 embedding-row lookup with optional scale ---------------------

void nnc_embed_row_bf16(float* y, const void* table, const int token_id,
                        const size_t n_embd, const float scale)
{
	const auto* row = static_cast<const uint16_t*>(table)
		+ static_cast<size_t>(token_id) * n_embd;
	// Inflate BF16 -> F32 then multiply by `scale`. Two-pass keeps the
	// SIMD code in nnc_bf16_to_f32_row simple.
	nnc_bf16_to_f32_row(row, y, n_embd);
	if (scale != 1.0f)
	{
		const __m256 vs = _mm256_set1_ps(scale);
		size_t i = 0;
		for (; i + 8 <= n_embd; i += 8)
			_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vs));
		for (; i < n_embd; ++i) y[i] *= scale;
	}
}

// ---- K-quant dequantizers ---------------------------------------------
//
// Layouts mirror ggml's `block_q*_K` structs (little-endian, x86-only).
// Block size is 256 elements = 8 sub-blocks of 32 (Q4_K/Q5_K) or 16
// sub-blocks of 16 (Q6_K). Decoders are scalar; intent is to dequant
// once at load time into BF16 / F32 storage that the existing gemv
// kernels can consume.

namespace
{
	struct block_q4_K
	{
		uint16_t d; // fp16, super-block scale-of-scales
		uint16_t dmin; // fp16, super-block scale-of-mins
		uint8_t scales[12]; // 8 sub-blocks * (6-bit scale + 6-bit min), packed
		uint8_t qs[128]; // 256 4-bit quants (low/high nibble)
	};
	static_assert(sizeof(block_q4_K) == 144, "Q4_K block layout mismatch");

	struct block_q5_K
	{
		uint16_t d;
		uint16_t dmin;
		uint8_t scales[12];
		uint8_t qh[32]; // 256 1-bit "high" quants (1 bit per element)
		uint8_t qs[128]; // 256 4-bit "low" quants
	};
	static_assert(sizeof(block_q5_K) == 176, "Q5_K block layout mismatch");

	struct block_q6_K
	{
		uint8_t ql[128]; // 256 4-bit "low" quants
		uint8_t qh[64]; // 256 2-bit "high" quants
		int8_t scales[16]; // 16 sub-block scales (i8)
		uint16_t d; // fp16 super-block scale
	};
	static_assert(sizeof(block_q6_K) == 210, "Q6_K block layout mismatch");

	// 6-bit scale/min unpacker shared by Q4_K and Q5_K.
	// `j` selects sub-block 0..7; the 12 scale bytes encode one 6-bit
	// scale + 6-bit min per sub-block via the canonical ggml packing.
	inline void get_scale_min_k4(const int j, const uint8_t* q,
	                             uint8_t& d, uint8_t& m)
	{
		if (j < 4)
		{
			d = q[j] & 63u;
			m = q[j + 4] & 63u;
		}
		else
		{
			d = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);
			m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
		}
	}
} // namespace

void nnc_dequantize_q4_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q4_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const float dmin = fp16_to_fp32_scalar(b[i].dmin);
		const uint8_t* q = b[i].qs;

		// 4 outer iterations of 64 outputs each (256 = 4*64).
		// Each iteration consumes 32 qs bytes and uses sub-blocks
		// `is` (low nibble) and `is+1` (high nibble), with `is` += 2.
		int is = 0;
		for (int j = 0; j < 256; j += 64)
		{
			uint8_t sc, mn;
			get_scale_min_k4(is + 0, b[i].scales, sc, mn);
			const float d1 = d * sc;
			const float m1 = dmin * mn;
			get_scale_min_k4(is + 1, b[i].scales, sc, mn);
			const float d2 = d * sc;
			const float m2 = dmin * mn;

			for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
			for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
			q += 32;
			is += 2;
		}
	}
}

void nnc_dequantize_q5_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q5_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const float dmin = fp16_to_fp32_scalar(b[i].dmin);
		const uint8_t* ql = b[i].qs;
		const uint8_t* qh = b[i].qh;

		int is = 0;
		uint8_t u1 = 1, u2 = 2;
		for (int j = 0; j < 256; j += 64)
		{
			uint8_t sc, mn;
			get_scale_min_k4(is + 0, b[i].scales, sc, mn);
			const float d1 = d * sc;
			const float m1 = dmin * mn;
			get_scale_min_k4(is + 1, b[i].scales, sc, mn);
			const float d2 = d * sc;
			const float m2 = dmin * mn;

			for (int l = 0; l < 32; ++l)
				*y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
			for (int l = 0; l < 32; ++l)
				*y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
			ql += 32;
			is += 2;
			u1 <<= 2;
			u2 <<= 2;
		}
	}
}

void nnc_dequantize_q6_k_to_f32(const void* blocks, float* dst,
                                const size_t n_elements)
{
	NNC_ASSERT(blocks && dst);
	NNC_ASSERT(n_elements > 0 && (n_elements % 256) == 0);

	const auto* b = static_cast<const block_q6_K*>(blocks);
	const size_t nb = n_elements / 256;
	float* y = dst;

	for (size_t i = 0; i < nb; ++i)
	{
		const float d = fp16_to_fp32_scalar(b[i].d);
		const uint8_t* ql = b[i].ql;
		const uint8_t* qh = b[i].qh;
		const int8_t* sc = b[i].scales;

		// 2 outer iterations of 128 outputs each.
		for (int n = 0; n < 256; n += 128)
		{
			for (int l = 0; l < 32; ++l)
			{
				const int is = l / 16;
				const int q1 = static_cast<int8_t>(
					(ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int q2 = static_cast<int8_t>(
					(ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int q3 = static_cast<int8_t>(
					(ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int q4 = static_cast<int8_t>(
					(ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l + 0] = d * sc[is + 0] * q1;
				y[l + 32] = d * sc[is + 2] * q2;
				y[l + 64] = d * sc[is + 4] * q3;
				y[l + 96] = d * sc[is + 6] * q4;
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

bool nnc_dequantize_kquant_to_f32(const uint32_t ggml_type, const void* blocks,
                                  float* dst, const size_t n_elements)
{
	switch (ggml_type)
	{
		case 12: nnc_dequantize_q4_k_to_f32(blocks, dst, n_elements);
			return true;
		case 13: nnc_dequantize_q5_k_to_f32(blocks, dst, n_elements);
			return true;
		case 14: nnc_dequantize_q6_k_to_f32(blocks, dst, n_elements);
			return true;
		default: return false;
	}
}

// ---- graph-level fusion: mul_mat -> repeat(bias) -> add ----------------

#include <unordered_map>

namespace
{
	std::unordered_map<const nnc_tensor*, const float*> g_fused_bias;
	std::unordered_map<const nnc_tensor*, void*> g_dst_override;
	std::unordered_map<const nnc_tensor*, bool> g_fused_gelu;

	bool is_1d_bias_for(const nnc_tensor* bias, const nnc_tensor* mm)
	{
		// 1D FP32 vector whose length matches the row count of the mul_mat
		// output (mm->ne[0]). The repeat broadcasts it to the full output.
		return bias && bias->n_dims == 1
			&& bias->type == NNC_TYPE_F32
			&& bias->ne[0] == mm->ne[0];
	}
}

void nnc_graph_prefuse(const nnc_cgraph* g)
{
	g_fused_bias.clear();
	g_dst_override.clear();
	g_fused_gelu.clear();
	if (!g) return;

	const int n = g->n_nodes;

	// Count how many nodes consume each tensor as a src. We only fuse if
	// the intermediate `repeat` and `add` outputs are referenced exactly
	// once — by the next node in the chain. Without this check, a future
	// graph that branches off the bias-added activation would silently
	// lose data when we set those nodes' op to NNC_OP_NONE.
	std::unordered_map<const nnc_tensor*, int> use_count;
	for (int j = 0; j < n; ++j)
	{
		const nnc_tensor* nd = g->nodes[j];
		if (nd->src0) ++use_count[nd->src0];
		if (nd->src1) ++use_count[nd->src1];
	}

	for (int i = 0; i + 2 < n; ++i)
	{
		const nnc_tensor* mm = g->nodes[i];
		nnc_tensor* rp = g->nodes[i + 1];
		nnc_tensor* ad = g->nodes[i + 2];

		if (mm->op != NNC_OP_MUL_MAT) continue;
		if (rp->op != NNC_OP_REPEAT) continue;
		if (ad->op != NNC_OP_ADD) continue;

		// repeat(bias) -> shape of mm
		if (rp->src1 != mm) continue; // repeat target shape must be mm
		if (!is_1d_bias_for(rp->src0, mm)) continue;

		// add(repeat, mm) or add(mm, repeat)
		const bool a = (ad->src0 == rp && ad->src1 == mm);
		const bool b = (ad->src0 == mm && ad->src1 == rp);
		if (!a && !b) continue;

		// Only fuse the FP16-weights / FP32-activations fast path that our
		// mul_mat shim already handles inline.
		if (!(mm->src0->type == NNC_TYPE_F16
			&& mm->src1->type == NNC_TYPE_F32))
			continue;

		// Refuse to fuse if `mm`, `rp`, or `ad` feed any other consumer.
		// `mm` is consumed by `rp`+`ad` (count 2), `rp` by `ad` (count 1).
		// Anything higher means the intermediate result is read elsewhere
		// and we can't safely zero out the producing op.
		auto count = [&](const nnc_tensor* t)
		{
			const auto it = use_count.find(t);
			return it == use_count.end() ? 0 : it->second;
		};
		if (count(mm) != 2 || count(rp) != 1) continue;

		g_fused_bias[mm] = static_cast<const float*>(rp->src0->data);
		g_dst_override[mm] = ad->data;
		rp->op = NNC_OP_NONE;
		ad->op = NNC_OP_NONE;

		if (i + 3 < n)
		{
			nnc_tensor* gl = g->nodes[i + 3];
			if (gl->op == NNC_OP_GELU
				&& gl->src0 == ad
				&& gl->type == NNC_TYPE_F32
				&& gl->ne[0] == ad->ne[0]
				&& count(ad) == 1)
			{
				g_dst_override[mm] = gl->data;
				g_fused_gelu[mm] = true;
				gl->op = NNC_OP_NONE;
			}
		}
	}
}

const float* nnc_fused_bias_for(const nnc_tensor* mm)
{
	const auto it = g_fused_bias.find(mm);
	return it == g_fused_bias.end() ? nullptr : it->second;
}

bool nnc_should_skip(const nnc_tensor* /*node*/)
{
	// Skipping is now done by mutating node->op to NNC_OP_NONE in prefuse;
	// keep this entry point for header compatibility / future use.
	return false;
}

void* nnc_fused_dst_for(const nnc_tensor* mm)
{
	const auto it = g_dst_override.find(mm);
	return it == g_dst_override.end() ? nullptr : it->second;
}

bool nnc_fused_gelu_for(const nnc_tensor* mm)
{
	return g_fused_gelu.contains(mm);
}
