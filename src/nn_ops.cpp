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
		using task_fn_t = void (*)(int tid, void* user);

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
			for (auto& w : workers_) w->thr.join();
		}

	private:
		struct slot
		{
			std::thread thr;
			std::atomic<unsigned> done{0};
		};

		static int decide_n_workers()
		{
			const unsigned hw = std::thread::hardware_concurrency();
			// Cap at 8 workers (9 threads total). For Gemma E2B the
			// per-token gemv compute is small enough that beyond ~8
			// threads we hit dispatch overhead and DRAM saturation.
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

	void bf16_gemv_worker(const int tid, void* u)
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
			nnc_gemv_bf16w_f32x_fn fn4;
			{
				std::lock_guard<std::mutex> lock(global_cache_mutex());
				fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
			}
			bf16_gemv_ctx ctx{
				static_cast<const uint8_t*>(W), x, y, cols,
				rows / 4, n_threads, fn4
			};
			pool.dispatch(&bf16_gemv_worker, &ctx);
			return;
		}

		nnc_gemv_bf16w_f32x_fn fn;
		{
			std::lock_guard<std::mutex> lock(global_cache_mutex());
			fn = global_cache().get_gemv_bf16w_f32x(rows, cols);
		}
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
		nnc_gemv_bf16w_f32x_fn fn4;
		{
			std::lock_guard<std::mutex> lock(global_cache_mutex());
			fn4 = global_cache().get_gemv_bf16w_f32x(4, cols);
		}
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

			pool.dispatch([](const int tid, void* u)
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
					if (scratch[0] > bv) { bv = scratch[0]; bi = base; }
					if (scratch[1] > bv) { bv = scratch[1]; bi = base + 1; }
					if (scratch[2] > bv) { bv = scratch[2]; bi = base + 2; }
					if (scratch[3] > bv) { bv = scratch[3]; bi = base + 3; }
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
			if (scratch[0] > bv) { bv = scratch[0]; best = static_cast<int>(r); }
			if (scratch[1] > bv) { bv = scratch[1]; best = static_cast<int>(r) + 1; }
			if (scratch[2] > bv) { bv = scratch[2]; best = static_cast<int>(r) + 2; }
			if (scratch[3] > bv) { bv = scratch[3]; best = static_cast<int>(r) + 3; }
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
		if (tmp[r] > bv) { bv = tmp[r]; best = static_cast<int>(r); }
	}
	return best;
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
                  const uint32_t n_rot, const int32_t pos, const float freq_base)
{
	if (n_rot == 0 || (n_rot & 1u) != 0) return; // n_rot must be even
	const uint32_t half = n_rot / 2;
	const float p = static_cast<float>(pos);
	for (uint32_t h = 0; h < n_heads; ++h)
	{
		float* xh = x + static_cast<size_t>(h) * head_dim;
		for (uint32_t i = 0; i < half; ++i)
		{
			const float inv = std::pow(freq_base,
			                           -static_cast<float>(2 * i) / static_cast<float>(n_rot));
			const float theta = p * inv;
			const float c = std::cos(theta);
			const float s = std::sin(theta);
			const float a = xh[i];
			const float b = xh[i + half];
			xh[i] = c * a - s * b;
			xh[i + half] = s * a + c * b;
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
