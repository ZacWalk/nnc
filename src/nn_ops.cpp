// nnc — neural-net op implementations.
// Hand-written AVX2/FMA/F16C SIMD kernels (gelu, softmax, layernorm,
// elementwise add) plus thin wrappers that route dot/gemv to JITted
// kernels from jit_kernel.cpp. Also hosts the small graph-level fuser
// that collapses mul_mat -> repeat(bias) -> add [-> gelu] chains.

#include "nn_ops.h"

#include "jit_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <mutex>

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
			// subnormal: normalize
			uint32_t mm = m;
			int32_t ee = -1;
			while ((mm & 0x400u) == 0)
			{
				mm <<= 1;
				--ee;
			}
			mm &= 0x3FFu;
			bits = (s << 31) | (static_cast<uint32_t>(127 - 15 + ee + 1) << 23) | (mm << 13);
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

	std::mutex& global_cache_mutex()
	{
		static std::mutex m;
		return m;
	}
}

float nnc_dot_f16_to_f32(const void* x, const void* y, const size_t n)
{
	if (n == 0) return 0.0f;

	if ((n & 31u) == 0 && n <= UINT32_MAX)
	{
		nnc_dot_f16_fn fn;
		{
			std::lock_guard<std::mutex> lock(global_cache_mutex());
			fn = global_cache().get_dot_f16(static_cast<uint32_t>(n));
		}
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

	// Pass 2: exp(p[i] - m), accumulate sum (scalar — AVX exp would need
	// a polynomial; keep it simple and correct for now).
	double sum = 0.0;
	for (size_t i = 0; i < n; ++i)
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

	// Pass 3: scale by 1/sum via AVX2.
	const float inv = static_cast<float>(1.0 / sum);
	const __m256 vinv = _mm256_set1_ps(inv);
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		_mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
	}
	for (; i < n; ++i) p[i] *= inv;
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

// ---- fused FP16 W * FP32 x -> FP32 y gemv ------------------------------

void nnc_gemv_f16w_f32x(const void* W, const float* x, float* y,
                        const uint32_t rows, const uint32_t cols)
{
	if (rows == 0 || cols == 0) return;

	if ((cols & 7u) == 0)
	{
		nnc_gemv_f16w_f32x_fn fn;
		{
			std::lock_guard<std::mutex> lock(global_cache_mutex());
			fn = global_cache().get_gemv_f16w_f32x(rows, cols);
		}
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

// ---- graph-level fusion: mul_mat -> repeat(bias) -> add ----------------

#include "runtime.h"

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
				&& gl->ne[0] == ad->ne[0])
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
