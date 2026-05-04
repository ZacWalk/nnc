// nnc — single test translation unit.
// All tests (app + jit) live here. Run with `nnc --test` (also -test, /test).
// Each test returns true on success. Exit code = number of failures.

#include "jit_buffer.h"
#include "jit_kernel.h"
#include "jit_ops.h"
#include "nn_ops.h"
#include "emitter_x64.h"
#include "runtime.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
	int g_pass = 0;
	int g_fail = 0;

	void report(const char* name, const bool ok, const char* msg = nullptr)
	{
		if (ok)
		{
			++g_pass;
			std::printf("[PASS] %s\n", name);
		}
		else
		{
			++g_fail;
			std::printf("[FAIL] %s%s%s\n", name, msg ? " : " : "", msg ? msg : "");
		}
	}

	bool close(const float a, const float b, const float rel_tol)
	{
		const float diff = std::fabs(a - b);
		const float scale = std::fmax(std::fabs(a), std::fabs(b));
		return diff <= rel_tol * std::fmax(scale, 1.0f);
	}

	// ---- CPU --------------------------------------------------------------

	bool test_cpu_has_avx2_fma()
	{
		const auto& f = nnc_cpu_features();
		return f.avx2 && f.fma;
	}

	// ---- JIT plumbing -----------------------------------------------------

	bool test_jit_return_42()
	{
		jit_buffer buf;
		x64_emitter e(buf);
		e.mov_r32_imm32(gpr::rax, 42);
		e.ret();
		using fn_t = int (*)();
		const auto fn = reinterpret_cast<fn_t>(buf.commit());
		return fn() == 42;
	}

	bool test_jit_add_two_ints()
	{
		jit_buffer buf;
		x64_emitter e(buf);
		e.emit_win64_arg_shuffle(2); // (a, b)
		e.lea_r32_base_index(gpr::rax, gpr::rcx, gpr::rdx);
		e.ret();
		using fn_t = int (*)(int, int);
		const auto fn = reinterpret_cast<fn_t>(buf.commit());
		return fn(2, 3) == 5 && fn(-1, 1) == 0 && fn(1000, 2345) == 3345;
	}

	// ---- dot_f32 ----------------------------------------------------------

	float scalar_dot(const float* a, const float* b, const size_t n)
	{
		double s = 0.0;
		for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
		return static_cast<float>(s);
	}

	bool test_jit_dot_f32()
	{
		jit_buffer buf;
		nnc_build_dot_f32(buf);
		const auto fn = reinterpret_cast<nnc_dot_f32_fn>(buf.commit());

		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		constexpr size_t sizes[] = {8, 64, 768, 4096};
		for (const size_t n : sizes)
		{
			std::vector<float> a(n), b(n);
			for (size_t i = 0; i < n; ++i)
			{
				a[i] = dist(rng);
				b[i] = dist(rng);
			}
			const float got = fn(a.data(), b.data(), n);
			const float ref = scalar_dot(a.data(), b.data(), n);
			if (!close(got, ref, 1e-4f))
			{
				std::printf("    n=%zu  jit=%g  ref=%g\n", n, got, ref);
				return false;
			}
		}
		return true;
	}

	// ---- gemv_f32 ---------------------------------------------------------

	void scalar_gemv(const float* W, const float* x, float* y, const uint32_t rows, const uint32_t cols)
	{
		for (uint32_t r = 0; r < rows; ++r)
		{
			double s = 0.0;
			for (uint32_t k = 0; k < cols; ++k) s += static_cast<double>(W[r * cols + k]) * x[k];
			y[r] = static_cast<float>(s);
		}
	}

	bool gemv_one(jit_kernel_cache& cache, const uint32_t rows, const uint32_t cols, std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<float> W(static_cast<size_t>(rows) * cols);
		std::vector<float> x(cols);
		std::vector<float> y_jit(rows), y_ref(rows);

		for (auto& v : W) v = dist(rng);
		for (auto& v : x) v = dist(rng);

		const auto fn = cache.get_gemv_f32(rows, cols);
		fn(W.data(), x.data(), y_jit.data());
		scalar_gemv(W.data(), x.data(), y_ref.data(), rows, cols);

		for (uint32_t r = 0; r < rows; ++r)
		{
			if (!close(y_jit[r], y_ref[r], 1e-4f))
			{
				std::printf("    gemv rows=%u cols=%u r=%u jit=%g ref=%g\n",
				            rows, cols, r, y_jit[r], y_ref[r]);
				return false;
			}
		}
		return true;
	}

	bool test_jit_gemv_f32()
	{
		jit_kernel_cache cache;
		std::mt19937 rng(0xBEEF);

		struct shape
		{
			uint32_t rows, cols;
		};
		constexpr shape shapes[] = {
			{1, 8},
			{4, 64},
			{16, 128},
			{32, 768},
		};
		for (const auto s : shapes)
		{
			if (!gemv_one(cache, s.rows, s.cols, rng)) return false;
		}
		return true;
	}

	bool test_jit_gemv_cache_reuse()
	{
		jit_kernel_cache cache;
		const auto a = cache.get_gemv_f32(16, 64);
		const auto b = cache.get_gemv_f32(16, 64);
		const auto c = cache.get_gemv_f32(16, 128); // different cols
		const bool ok = (a == b) && (a != c) && (cache.size() == 2);
		if (!ok)
		{
			std::printf("    cache: a=%p b=%p c=%p size=%zu\n", a, b, c, cache.size());
		}
		return ok;
	}

	// ---- nn_ops: gelu (Stage A) -------------------------------------------

	float ref_gelu_scalar(const float v)
	{
		constexpr float SQRT_2_OVER_PI = 0.79788456080286535f;
		constexpr float A = 0.044715f;
		const float u = SQRT_2_OVER_PI * (v + A * v * v * v);
		return 0.5f * v * (1.0f + std::tanh(u));
	}

	bool test_nnc_gelu_f32()
	{
		std::mt19937 rng(0xDEAD);
		std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
		constexpr size_t n = 1024;
		std::vector<float> x(n), y(n);
		for (auto& v : x) v = dist(rng);

		nnc_gelu_f32(y.data(), x.data(), n);

		for (size_t i = 0; i < n; ++i)
		{
			const float ref = ref_gelu_scalar(x[i]);
			if (!close(y[i], ref, 5e-4f))
			{
				std::printf("    gelu i=%zu x=%g y=%g ref=%g\n", i, x[i], y[i], ref);
				return false;
			}
		}

		// Spot-check known values.
		constexpr float spot[] = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f};
		float out[5];
		nnc_gelu_f32(out, spot, 5);
		if (!close(out[0], 0.0f, 1e-6f)) return false;
		if (!close(out[1], 0.841192f, 1e-4f)) return false; // GELU(1)  ~ 0.8412
		if (!close(out[2], -0.158808f, 1e-4f)) return false; // GELU(-1) ~ -0.1588
		return true;
	}

	// ---- nn_ops: dot_f16_to_f32 (Stage B) ---------------------------------

	// IEEE 754 binary32 -> binary16, round-to-nearest-even.
	uint16_t fp32_to_fp16(const float f)
	{
		uint32_t bits;
		std::memcpy(&bits, &f, 4);
		const uint32_t s = (bits >> 31) & 0x1u;
		int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
		const uint32_t m = bits & 0x7FFFFFu;

		if (e >= 31) return static_cast<uint16_t>((s << 15) | (0x1Fu << 10)); // overflow -> inf
		if (e <= 0) return static_cast<uint16_t>(s << 15); // underflow -> 0
		// round-to-nearest-even on the 13 dropped bits
		const uint32_t round = (m >> 12) & 1u;
		const uint32_t sticky = (m & 0xFFFu) ? 1u : 0u;
		uint32_t mh = (m >> 13) + (round & (sticky | ((m >> 13) & 1u)));
		if (mh & 0x400u)
		{
			mh = 0;
			++e;
			if (e >= 31) return static_cast<uint16_t>((s << 15) | (0x1Fu << 10));
		}
		return static_cast<uint16_t>((s << 15) | (static_cast<uint32_t>(e) << 10) | (mh & 0x3FFu));
	}

	float fp16_to_fp32(const uint16_t h)
	{
		const uint32_t s = (h >> 15) & 0x1u;
		const uint32_t e = (h >> 10) & 0x1Fu;
		const uint32_t m = h & 0x3FFu;
		uint32_t bits;
		if (e == 0) bits = s << 31;
		else if (e == 31) bits = (s << 31) | (0xFFu << 23) | (m << 13);
		else bits = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
		float f;
		std::memcpy(&f, &bits, 4);
		return f;
	}

	bool dot_f16_one(const size_t n, std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::vector<uint16_t> x(n), y(n);
		for (size_t i = 0; i < n; ++i)
		{
			x[i] = fp32_to_fp16(dist(rng));
			y[i] = fp32_to_fp16(dist(rng));
		}

		// Reference: convert halves back to floats and accumulate in double.
		double ref = 0.0;
		for (size_t i = 0; i < n; ++i)
		{
			ref += static_cast<double>(fp16_to_fp32(x[i]))
				* static_cast<double>(fp16_to_fp32(y[i]));
		}

		const float got = nnc_dot_f16_to_f32(x.data(), y.data(), n);
		const float tol = 1e-3f * std::max(1.0f, std::fabs(static_cast<float>(ref)));
		if (std::fabs(got - static_cast<float>(ref)) > tol)
		{
			std::printf("    dot_f16 n=%zu got=%g ref=%g diff=%g tol=%g\n",
			            n, got, static_cast<double>(ref),
			            std::fabs(got - ref), tol);
			return false;
		}
		return true;
	}

	bool test_nnc_dot_f16_to_f32()
	{
		std::mt19937 rng(0xF16C);
		// JIT path sizes (multiples of 32) — covers GPT-2 117M shapes 64, 768, 3072
		// plus a few corners.
		for (const size_t n : {32u, 64u, 256u, 768u, 1024u, 3072u, 4096u})
		{
			if (!dot_f16_one(n, rng)) return false;
		}
		// Scalar fallback path (not multiple of 32).
		for (const size_t n : {1u, 7u, 33u, 100u})
		{
			if (!dot_f16_one(n, rng)) return false;
		}
		return true;
	}

	// ---- nn_ops: softmax + layernorm (Stage C) -----------------------------

	bool test_nnc_softmax_f32()
	{
		// Known-good: softmax([1,2,3,4]) -> exp normalized.
		float p[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		nnc_softmax_f32_inplace(p, 4);

		float e[4];
		double s = 0;
		for (int i = 0; i < 4; ++i)
		{
			e[i] = std::exp(static_cast<float>(i + 1) - 4.0f);
			s += e[i];
		}
		for (int i = 0; i < 4; ++i)
		{
			const float ref = static_cast<float>(e[i] / s);
			if (!close(p[i], ref, 1e-6f))
			{
				std::printf("    softmax i=%d got=%g ref=%g\n", i, p[i], ref);
				return false;
			}
		}

		// -INFINITY masks: those entries must become 0 and the rest sum to 1.
		float q[5] = {0.5f, -INFINITY, 1.5f, -INFINITY, 2.5f};
		nnc_softmax_f32_inplace(q, 5);
		if (q[1] != 0.0f || q[3] != 0.0f) return false;
		const float total = q[0] + q[1] + q[2] + q[3] + q[4];
		if (!close(total, 1.0f, 1e-6f)) return false;
		// Strictly increasing for the unmasked entries.
		if (!(q[0] < q[2] && q[2] < q[4])) return false;

		// Random vector: result sums to 1, all >= 0.
		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
		constexpr size_t n = 1024;
		std::vector<float> v(n);
		for (auto& x : v) x = dist(rng);
		nnc_softmax_f32_inplace(v.data(), n);
		double sum = 0;
		for (const auto x : v)
		{
			if (x < 0.0f) return false;
			sum += x;
		}
		if (std::fabs(sum - 1.0) > 1e-5)
		{
			std::printf("    softmax sum=%g\n", sum);
			return false;
		}
		return true;
	}

	bool test_nnc_layernorm_f32()
	{
		std::mt19937 rng(0xBEEF);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		constexpr size_t n = 768;
		std::vector<float> x(n), y(n);
		for (auto& v : x) v = dist(rng);

		nnc_layernorm_f32(y.data(), x.data(), n, 1e-5f);

		double mean = 0, var = 0;
		for (const auto v : y) mean += v;
		mean /= n;
		for (const auto v : y) var += (v - mean) * (v - mean);
		var /= n;
		if (std::fabs(mean) > 1e-5)
		{
			std::printf("    norm mean=%g\n", mean);
			return false;
		}
		if (std::fabs(var - 1.0) > 1e-3)
		{
			std::printf("    norm var=%g\n", var);
			return false;
		}

		// Aliasing: y == x must work (in-place).
		std::vector<float> z = x;
		nnc_layernorm_f32(z.data(), z.data(), n, 1e-5f);
		for (size_t i = 0; i < n; ++i)
		{
			if (!close(z[i], y[i], 1e-5f))
			{
				std::printf("    norm alias i=%zu z=%g y=%g\n", i, z[i], y[i]);
				return false;
			}
		}
		return true;
	}

	bool test_nnc_rmsnorm_f32()
	{
		std::mt19937 rng(0xCAFE);
		std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
		// Cover Gemma E2B/E4B hidden sizes plus a small odd size for the tail.
		constexpr size_t sizes[] = {1536, 2560, 256, 13};
		for (const size_t n : sizes)
		{
			std::vector<float> x(n), y(n);
			for (auto& v : x) v = dist(rng);

			nnc_rmsnorm_f32(y.data(), x.data(), n, 1e-6f);

			// Scalar reference.
			double s2 = 0.0;
			for (const auto v : x) s2 += static_cast<double>(v) * v;
			const float scale = static_cast<float>(1.0 / std::sqrt(s2 / static_cast<double>(n) + 1e-6));
			for (size_t i = 0; i < n; ++i)
			{
				const float ref = x[i] * scale;
				if (!close(y[i], ref, 1e-5f))
				{
					std::printf("    rmsnorm n=%zu i=%zu y=%g ref=%g\n", n, i, y[i], ref);
					return false;
				}
			}

			// rms(y) should be ~1.
			double sy2 = 0.0;
			for (const auto v : y) sy2 += static_cast<double>(v) * v;
			const double rms_y = std::sqrt(sy2 / static_cast<double>(n));
			if (std::fabs(rms_y - 1.0) > 1e-3)
			{
				std::printf("    rmsnorm n=%zu rms(y)=%g\n", n, rms_y);
				return false;
			}

			// Aliasing.
			std::vector<float> z = x;
			nnc_rmsnorm_f32(z.data(), z.data(), n, 1e-6f);
			for (size_t i = 0; i < n; ++i)
				if (!close(z[i], y[i], 1e-5f)) return false;
		}
		return true;
	}

	// ---- nn_ops: gemv_f16w_f32x (Stage D — fused mul_mat) -----------------

	bool gemv_f16w_one(const uint32_t rows, const uint32_t cols, std::mt19937& rng)
	{
		std::uniform_real_distribution<float> wdist(-0.5f, 0.5f);
		std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

		std::vector<uint16_t> W(static_cast<size_t>(rows) * cols);
		std::vector<float> x(cols);
		std::vector<float> y(rows, 0.0f);

		for (auto& w : W) w = fp32_to_fp16(wdist(rng));
		for (auto& v : x) v = xdist(rng);

		nnc_gemv_f16w_f32x(W.data(), x.data(), y.data(), rows, cols);

		// Reference: same algorithm, double accumulation.
		for (uint32_t r = 0; r < rows; ++r)
		{
			double s = 0.0;
			const uint16_t* row = W.data() + static_cast<size_t>(r) * cols;
			for (uint32_t k = 0; k < cols; ++k)
			{
				s += static_cast<double>(fp16_to_fp32(row[k])) * x[k];
			}
			const float ref = static_cast<float>(s);
			const float tol = 1e-3f * std::max(1.0f, std::fabs(ref));
			if (std::fabs(y[r] - ref) > tol)
			{
				std::printf("    gemv_f16w r=%u rows=%u cols=%u got=%g ref=%g diff=%g tol=%g\n",
				            r, rows, cols, y[r], ref,
				            std::fabs(y[r] - ref), tol);
				return false;
			}
		}
		return true;
	}

	bool test_nnc_gemv_f16w_f32x()
	{
		std::mt19937 rng(0x6EF5);
		// JIT path (cols multiple of 8). Covers GPT-2 117M shapes:
		//   768 -> 768 (Q,K,V proj, FFN_out output)
		//   768 -> 3072 (FFN_in)
		//   3072 -> 768 (FFN_out)
		//   768 -> 50257 (vocab projection — large rows, exercises rel32)
		struct sh
		{
			uint32_t rows, cols;
		};
		const sh shapes[] = {
			{1, 8}, {4, 64}, {16, 128},
			{768, 768}, {3072, 768}, {768, 3072},
			{50257, 768},
		};
		for (const auto s : shapes)
		{
			if (!gemv_f16w_one(s.rows, s.cols, rng)) return false;
		}
		// Scalar fallback (cols not a multiple of 8).
		if (!gemv_f16w_one(7, 5, rng)) return false;
		return true;
	}

	// ---- BF16 -------------------------------------------------------------

	bool test_bf16_round_trip()
	{
		// Round-trip a handful of representative values; values with no
		// fractional bits beyond bit 16 must round-trip exactly.
		constexpr float exact[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 1024.0f, -2048.0f};
		for (const float v : exact)
		{
			const float r = nnc_bf16_to_f32(nnc_f32_to_bf16(v));
			if (r != v) return false;
		}
		// Generic values: |relerr| <= 2^-7 (BF16 has 7 mantissa bits + 1 hidden).
		std::mt19937 rng(0xBF16);
		std::uniform_real_distribution<float> d(-3.0f, 3.0f);
		for (int i = 0; i < 1024; ++i)
		{
			const float v = d(rng);
			const float r = nnc_bf16_to_f32(nnc_f32_to_bf16(v));
			const float err = std::fabs(r - v);
			const float scale = std::fmax(std::fabs(v), 1.0f);
			if (err / scale > (1.0f / 128.0f)) return false;
		}
		return true;
	}

	bool test_bf16_to_f32_row()
	{
		// Compare AVX2 batched conversion to the scalar reference.
		std::mt19937 rng(0xBF16ABCD);
		std::uniform_real_distribution<float> d(-100.0f, 100.0f);
		constexpr size_t N = 137; // not a multiple of 8 -> exercise tail
		std::vector<nnc_bf16_t> src(N);
		for (size_t i = 0; i < N; ++i)
			src[i] = nnc_f32_to_bf16(d(rng));
		std::vector<float> got(N), ref(N);
		nnc_bf16_to_f32_row(src.data(), got.data(), N);
		for (size_t i = 0; i < N; ++i)
			ref[i] = nnc_bf16_to_f32(src[i]);
		for (size_t i = 0; i < N; ++i)
			if (got[i] != ref[i]) return false;
		return true;
	}

	// ---- nn_ops: gemv_bf16w_f32x ------------------------------------------

	bool test_nnc_gemv_bf16w_f32x()
	{
		std::mt19937 rng(0xB16E);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// E2B/E4B Gemma shapes plus odd small ones for the tail path.
		const sh shapes[] = {
			{1, 8}, {7, 5}, {16, 1536},
			{1536, 1536}, {2048, 1536}, {6144, 1536},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			std::vector<float> Wf32(W.size());
			for (size_t i = 0; i < W.size(); ++i)
			{
				const float v = d(rng);
				W[i] = nnc_f32_to_bf16(v);
				Wf32[i] = nnc_bf16_to_f32(W[i]);
			}
			std::vector<float> x(s.cols), y(s.rows), ref(s.rows);
			for (auto& v : x) v = d(rng);

			nnc_gemv_bf16w_f32x(W.data(), x.data(), y.data(), s.rows, s.cols);

			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				const float* row = Wf32.data() + static_cast<size_t>(r) * s.cols;
				for (uint32_t k = 0; k < s.cols; ++k)
					acc += static_cast<double>(row[k]) * x[k];
				ref[r] = static_cast<float>(acc);
			}
			// FMA reorders rounding vs scalar; allow modest relative error.
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				if (!close(y[r], ref[r], 1e-3f))
				{
					std::printf("    gemv_bf16w r=%u y=%g ref=%g (rows=%u cols=%u)\n",
					            r, y[r], ref[r], s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: gemv_bf16w_argmax_f32x -----------------------------------

	bool test_nnc_gemv_bf16w_argmax_f32x()
	{
		std::mt19937 rng(0xA12A);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// Cover the fast path (rows%4==0, cols%8==0) plus the fallback.
		const sh shapes[] = {
			{4, 32}, {16, 256}, {2048, 256}, {256, 1536}, {7, 5},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			std::vector<float> Wf32(W.size());
			for (size_t i = 0; i < W.size(); ++i)
			{
				const float v = d(rng);
				W[i] = nnc_f32_to_bf16(v);
				Wf32[i] = nnc_bf16_to_f32(W[i]);
			}
			std::vector<float> x(s.cols);
			for (auto& v : x) v = d(rng);

			// Reference: full materialise then linear scan, BF16-accurate.
			std::vector<float> ref(s.rows);
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				const float* row = Wf32.data() + static_cast<size_t>(r) * s.cols;
				for (uint32_t k = 0; k < s.cols; ++k)
					acc += static_cast<double>(row[k]) * x[k];
				ref[r] = static_cast<float>(acc);
			}
			int ref_best = 0;
			for (uint32_t r = 1; r < s.rows; ++r)
				if (ref[r] > ref[ref_best]) ref_best = static_cast<int>(r);

			const int got = nnc_gemv_bf16w_argmax_f32x(W.data(), x.data(),
			                                           s.rows, s.cols);
			// FMA reorder may flip ties; if got != ref_best require their
			// values to be within tolerance.
			if (got != ref_best)
			{
				const float diff = std::fabs(ref[got] - ref[ref_best]);
				const float tol = 1e-3f * std::max(1.0f, std::fabs(ref[ref_best]));
				if (diff > tol)
				{
					std::printf("    argmax got=%d ref=%d ref_v=%g got_v=%g (rows=%u cols=%u)\n",
					            got, ref_best, ref[ref_best], ref[got],
					            s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: q8_0 quantize roundtrip + gemv ---------------------------

	bool test_nnc_q8_0_quantize_roundtrip()
	{
		// Quantize a synthetic BF16 row and verify the recovered (scale*q)
		// values are within absmax/127 of the original — i.e. the worst
		// per-element error is bounded by the per-block step size.
		std::mt19937 rng(0xC0FFEE);
		std::uniform_real_distribution<float> d(-2.5f, 2.5f);
		constexpr size_t rows = 3, cols = 64;
		std::vector<nnc_bf16_t> W(rows * cols);
		std::vector<float> Wf(rows * cols);
		for (size_t i = 0; i < W.size(); ++i)
		{
			Wf[i] = d(rng);
			W[i] = nnc_f32_to_bf16(Wf[i]);
			Wf[i] = nnc_bf16_to_f32(W[i]); // re-quantize for fair compare
		}
		std::vector<int8_t> qs(rows * cols);
		std::vector<float> scales(rows * (cols / 32));
		nnc_quantize_bf16_to_q8_0(W.data(), qs.data(), scales.data(), rows, cols);

		for (size_t r = 0; r < rows; ++r)
		{
			for (size_t b = 0; b < cols / 32; ++b)
			{
				const float scale = scales[r * (cols / 32) + b];
				// Step size = scale; allow one step of error.
				for (size_t k = 0; k < 32; ++k)
				{
					const float got = scale * static_cast<float>(qs[r * cols + b * 32 + k]);
					const float ref = Wf[r * cols + b * 32 + k];
					if (std::fabs(got - ref) > scale + 1e-6f)
					{
						std::printf("    q8_0 roundtrip r=%zu b=%zu k=%zu got=%g ref=%g scale=%g\n",
						            r, b, k, got, ref, scale);
						return false;
					}
				}
			}
		}
		return true;
	}

	bool test_nnc_gemv_q8_0_f32x()
	{
		// Quantize a BF16 weight matrix, run JIT Q8_0 gemv, and compare
		// to a scalar reference computed from the SAME quantized weights
		// (so the only error source is FMA reorder vs scalar order).
		std::mt19937 rng(0xC8E2);
		std::uniform_real_distribution<float> d(-1.0f, 1.0f);
		struct sh
		{
			uint32_t rows, cols;
		};
		// cover small (single-thread) and parallel paths.
		const sh shapes[] = {
			{1, 32}, {8, 64}, {64, 128},
			{256, 256}, {512, 512}, {2048, 2048},
		};
		for (const auto s : shapes)
		{
			std::vector<nnc_bf16_t> W(static_cast<size_t>(s.rows) * s.cols);
			for (auto& w : W) w = nnc_f32_to_bf16(d(rng));

			std::vector<int8_t> qs(W.size());
			std::vector<float> scales(static_cast<size_t>(s.rows) * (s.cols / 32));
			nnc_quantize_bf16_to_q8_0(W.data(), qs.data(), scales.data(),
			                          s.rows, s.cols);

			std::vector<float> x(s.cols), y(s.rows), ref(s.rows);
			for (auto& v : x) v = d(rng);

			nnc_gemv_q8_0_f32x(qs.data(), scales.data(), x.data(), y.data(),
			                   s.rows, s.cols);

			// Reference: dequantize then dot, in double precision.
			const size_t scale_stride = s.cols / 32;
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				double acc = 0.0;
				for (uint32_t b = 0; b < scale_stride; ++b)
				{
					const float scale = scales[r * scale_stride + b];
					for (uint32_t k = 0; k < 32; ++k)
					{
						const float w = scale * qs[r * s.cols + b * 32 + k];
						acc += static_cast<double>(w) * x[b * 32 + k];
					}
				}
				ref[r] = static_cast<float>(acc);
			}
			for (uint32_t r = 0; r < s.rows; ++r)
			{
				const float tol = 1e-3f * std::max(1.0f, std::fabs(ref[r]));
				if (std::fabs(y[r] - ref[r]) > tol)
				{
					std::printf("    q8_0 gemv r=%u y=%g ref=%g (rows=%u cols=%u)\n",
					            r, y[r], ref[r], s.rows, s.cols);
					return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: K-quant dequantizers ------------------------------------

	// Helper: build a Q4_K block where every super-scale = 1, every
	// sub-block scale = 1 and min = 0, so dequant(qs) == per-nibble
	// integer values. Returns a 144-byte block image.
	void build_q4k_unit_block(uint8_t out[144], const uint8_t qs[128])
	{
		std::memset(out, 0, 144);
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(out + 0, &one_h, 2); // d
		// dmin = 0 (already memset)
		uint8_t* sc = out + 4;
		// sub-blocks 0..3: scale in low 6 of sc[0..3]; min = 0 in sc[4..7].
		sc[0] = 1; sc[1] = 1; sc[2] = 1; sc[3] = 1;
		// sub-blocks 4..7: scale low 4 in sc[8..11], min in sc[8..11]>>4.
		// High 2 bits of scale come from sc[0..3]>>6 (=0). Want scale=1.
		sc[8] = 0x01; sc[9] = 0x01; sc[10] = 0x01; sc[11] = 0x01;
		std::memcpy(out + 16, qs, 128);
	}

	bool test_nnc_dequantize_q4_k()
	{
		// qs[i] = 0x21 for all i: low nibble = 1, high nibble = 2.
		uint8_t qs[128];
		std::memset(qs, 0x21, sizeof(qs));
		uint8_t blk[144];
		build_q4k_unit_block(blk, qs);

		float out[256];
		nnc_dequantize_q4_k_to_f32(blk, out, 256);

		// Each of the 4 outer iterations writes 64 outputs:
		// 32 from low nibble (=1), 32 from high nibble (=2).
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k)
				if (out[j * 64 + k] != 1.0f) return false;
			for (int k = 0; k < 32; ++k)
				if (out[j * 64 + 32 + k] != 2.0f) return false;
		}
		return true;
	}

	bool test_nnc_dequantize_q5_k()
	{
		// Same scale/min layout as Q4_K, plus qh: 1 bit per element
		// across all 4 outer iterations (u1,u2 walk bits 0/1, 2/3, 4/5, 6/7).
		uint8_t blk[176];
		std::memset(blk, 0, sizeof(blk));
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(blk + 0, &one_h, 2); // d
		uint8_t* sc = blk + 4;
		sc[0] = sc[1] = sc[2] = sc[3] = 1;
		sc[8] = sc[9] = sc[10] = sc[11] = 0x01;
		uint8_t* qh = blk + 16;
		uint8_t* qs = blk + 16 + 32;

		// Case 1: qh all zero → no +16, output identical to Q4_K test.
		std::memset(qh, 0x00, 32);
		std::memset(qs, 0x21, 128);

		float out[256];
		nnc_dequantize_q5_k_to_f32(blk, out, 256);
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k) if (out[j * 64 + k] != 1.0f) return false;
			for (int k = 0; k < 32; ++k) if (out[j * 64 + 32 + k] != 2.0f) return false;
		}

		// Case 2: qh all 0xFF → every element gets +16 added.
		std::memset(qh, 0xFF, 32);
		nnc_dequantize_q5_k_to_f32(blk, out, 256);
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 32; ++k) if (out[j * 64 + k] != 17.0f) return false;
			for (int k = 0; k < 32; ++k) if (out[j * 64 + 32 + k] != 18.0f) return false;
		}
		return true;
	}

	bool test_fp16_subnormal_dequant()
	{
		// Regression: fp16 subnormal decode previously had an off-by-one
		// that produced values 2x too small. K-quant `d` super-block scales
		// frequently land in subnormal range, biasing dequanted weights.
		//
		// Build a Q4_K block with d = smallest subnormal half (0x0001
		// = 2^-24), dmin = 0, sub-block scale[0] = 1, qs low nibble = 1.
		// Then y[0] should be 1 * d * 1 - 0 = 2^-24 ≈ 5.96e-8.
		uint8_t blk[144];
		std::memset(blk, 0, sizeof(blk));
		const uint16_t d_smallest_subnormal = 0x0001; // 2^-24
		std::memcpy(blk + 0, &d_smallest_subnormal, 2);
		// dmin = 0 (already)
		blk[4] = 1; // scales[0] = sc=1, mn=0
		blk[8] = 0; // scales[8..11] = 0 (unused for j<4)
		std::memset(blk + 16, 0x11, 128); // qs: low=1, high=1

		float out[256];
		nnc_dequantize_q4_k_to_f32(blk, out, 256);
		const float expected = std::ldexp(1.0f, -24); // 2^-24
		// Tight relative tolerance — decode should be exact.
		const float err = std::fabs(out[0] - expected);
		return err < 1e-12f;
	}

	bool test_nnc_dequantize_q6_k()
	{
		// d = 1, all 16 sub-block scales = 1, ql = 0x10 (low nibble 0,
		// high nibble 1), qh = 0 (top 2 bits of every quant = 0). Then
		// q1 = q2 = 0 - 32 = -32; q3 = q4 = 1 - 32 = -31.
		uint8_t blk[210];
		std::memset(blk, 0, sizeof(blk));
		std::memset(blk + 0, 0x10, 128); // ql
		// qh already zeroed
		std::memset(blk + 128 + 64, 1, 16); // scales (i8) = 1
		const uint16_t one_h = fp32_to_fp16(1.0f);
		std::memcpy(blk + 208, &one_h, 2); // d

		float out[256];
		nnc_dequantize_q6_k_to_f32(blk, out, 256);

		// For each of 2 outer iters writing 128 outputs:
		//   y[l +  0] = -32, y[l + 32] = -32, y[l + 64] = -31, y[l + 96] = -31
		for (int n = 0; n < 256; n += 128)
		{
			for (int l = 0; l < 32; ++l)
			{
				if (out[n + l + 0] != -32.0f) return false;
				if (out[n + l + 32] != -32.0f) return false;
				if (out[n + l + 64] != -31.0f) return false;
				if (out[n + l + 96] != -31.0f) return false;
			}
		}
		return true;
	}

	// ---- nn_ops: swiglu ---------------------------------------------------

	bool test_nnc_swiglu_f32()
	{
		std::mt19937 rng(0x5167);
		std::uniform_real_distribution<float> d(-3.0f, 3.0f);
		constexpr size_t N = 257;
		std::vector<float> g(N), u(N), y(N);
		for (size_t i = 0; i < N; ++i)
		{
			g[i] = d(rng);
			u[i] = d(rng);
		}

		nnc_swiglu_f32(y.data(), g.data(), u.data(), N);

		for (size_t i = 0; i < N; ++i)
		{
			const float sig = 1.0f / (1.0f + std::exp(-g[i]));
			const float ref = g[i] * sig * u[i];
			if (!close(y[i], ref, 1e-5f)) return false;
		}
		// Aliasing: y == gate must work.
		std::vector<float> z = g;
		nnc_swiglu_f32(z.data(), z.data(), u.data(), N);
		for (size_t i = 0; i < N; ++i)
			if (!close(z[i], y[i], 1e-5f)) return false;
		return true;
	}

	// ---- nn_ops: rope -----------------------------------------------------

	bool test_nnc_rope_f32()
	{
		// Property check: pos=0 must be the identity.
		{
			std::mt19937 rng(0x5051);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 8, D = 64, R = 64;
			std::vector<float> x(H * D), x0(H * D);
			for (auto& v : x) v = d(rng);
			x0 = x;
			nnc_rope_f32(x.data(), H, D, R, 0, 10000.0f);
			for (size_t i = 0; i < x.size(); ++i)
				if (!close(x[i], x0[i], 1e-6f)) return false;
		}

		// Numerical equivalence vs scalar reference.
		{
			std::mt19937 rng(0x5052);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 4, D = 64, R = 32; // R < D: trailing lanes pass through
			std::vector<float> x(H * D), ref(H * D);
			for (auto& v : x) v = d(rng);
			ref = x;

			constexpr int32_t pos = 17;
			constexpr float base = 1e6f;
			nnc_rope_f32(x.data(), H, D, R, pos, base);

			constexpr uint32_t half = R / 2;
			for (uint32_t h = 0; h < H; ++h)
			{
				float* rh = ref.data() + h * D;
				for (uint32_t i = 0; i < half; ++i)
				{
					const float inv = std::pow(base, -static_cast<float>(2 * i) / static_cast<float>(R));
					const float theta = pos * inv;
					const float c = std::cos(theta), s = std::sin(theta);
					const float a = rh[i], b = rh[i + half];
					rh[i] = c * a - s * b;
					rh[i + half] = s * a + c * b;
				}
			}
			for (size_t i = 0; i < x.size(); ++i)
				if (!close(x[i], ref[i], 1e-5f)) return false;
		}

		// Property: rotation preserves per-pair magnitude.
		{
			std::mt19937 rng(0x5053);
			std::uniform_real_distribution<float> d(-1.0f, 1.0f);
			constexpr uint32_t H = 2, D = 16, R = 16;
			std::vector<float> x(H * D), x0(H * D);
			for (auto& v : x) v = d(rng);
			x0 = x;
			nnc_rope_f32(x.data(), H, D, R, 42, 10000.0f);
			for (uint32_t h = 0; h < H; ++h)
			{
				const float* a = x0.data() + h * D;
				const float* b = x.data() + h * D;
				constexpr uint32_t half = R / 2;
				for (uint32_t i = 0; i < half; ++i)
				{
					const float m0 = a[i] * a[i] + a[i + half] * a[i + half];
					const float m1 = b[i] * b[i] + b[i + half] * b[i + half];
					if (std::fabs(m0 - m1) > 1e-4f * std::fmax(m0, 1e-6f)) return false;
				}
			}
		}
		return true;
	}

	// ---- nn_ops: softcap --------------------------------------------------

	bool test_nnc_softcap_f32()
	{
		constexpr float cap = 30.0f;
		constexpr float in[] = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 30.0f, 100.0f, -100.0f, 1e6f};
		constexpr size_t N = sizeof(in) / sizeof(in[0]);
		float out[N];
		nnc_softcap_f32(out, in, N, cap);
		for (size_t i = 0; i < N; ++i)
		{
			const float ref = std::tanh(in[i] / cap) * cap;
			if (!close(out[i], ref, 1e-5f)) return false;
			if (std::fabs(out[i]) > cap + 1e-4f) return false;
		}
		// Soft-cap of 0 stays 0.
		if (out[0] != 0.0f) return false;
		// Saturates near +/- cap for large inputs.
		if (out[8] < cap - 1e-3f || out[8] > cap) return false;
		return true;
	}

	// ---- nn_ops: embed_row_bf16 ------------------------------------------

	bool test_nnc_embed_row_bf16()
	{
		constexpr size_t n_embd = 17;
		constexpr size_t n_vocab = 5;
		std::vector<nnc_bf16_t> table(n_vocab * n_embd);
		std::mt19937 rng(0xE3B0);
		std::uniform_real_distribution<float> d(-2.0f, 2.0f);
		for (auto& v : table) v = nnc_f32_to_bf16(d(rng));

		constexpr int tok = 3;
		constexpr float scale = 1.5f;
		std::vector<float> y(n_embd);
		nnc_embed_row_bf16(y.data(), table.data(), tok, n_embd, scale);
		for (size_t i = 0; i < n_embd; ++i)
		{
			const float ref = nnc_bf16_to_f32(table[tok * n_embd + i]) * scale;
			if (!close(y[i], ref, 1e-6f)) return false;
		}
		return true;
	}
}

int run_tests()
{
	std::printf("nnc: running tests\n");

	report("cpu_has_avx2_fma", test_cpu_has_avx2_fma());
	report("jit_return_42", test_jit_return_42());
	report("jit_add_two_ints", test_jit_add_two_ints());
	report("jit_dot_f32", test_jit_dot_f32());
	report("jit_gemv_f32", test_jit_gemv_f32());
	report("jit_gemv_cache_reuse", test_jit_gemv_cache_reuse());
	report("nnc_gelu_f32", test_nnc_gelu_f32());
	report("nnc_dot_f16_to_f32", test_nnc_dot_f16_to_f32());
	report("nnc_softmax_f32", test_nnc_softmax_f32());
	report("nnc_layernorm_f32", test_nnc_layernorm_f32());
	report("nnc_rmsnorm_f32", test_nnc_rmsnorm_f32());
	report("nnc_gemv_f16w_f32x", test_nnc_gemv_f16w_f32x());
	report("bf16_round_trip", test_bf16_round_trip());
	report("bf16_to_f32_row", test_bf16_to_f32_row());
	report("nnc_gemv_bf16w_f32x", test_nnc_gemv_bf16w_f32x());
	report("nnc_gemv_bf16w_argmax_f32x", test_nnc_gemv_bf16w_argmax_f32x());
	report("nnc_q8_0_quantize_roundtrip", test_nnc_q8_0_quantize_roundtrip());
	report("nnc_gemv_q8_0_f32x", test_nnc_gemv_q8_0_f32x());
	report("nnc_dequantize_q4_k", test_nnc_dequantize_q4_k());
	report("nnc_dequantize_q5_k", test_nnc_dequantize_q5_k());
	report("fp16_subnormal_dequant", test_fp16_subnormal_dequant());
	report("nnc_dequantize_q6_k", test_nnc_dequantize_q6_k());
	report("nnc_swiglu_f32", test_nnc_swiglu_f32());
	report("nnc_rope_f32", test_nnc_rope_f32());
	report("nnc_softcap_f32", test_nnc_softcap_f32());
	report("nnc_embed_row_bf16", test_nnc_embed_row_bf16());

	std::printf("nnc: %d passed, %d failed\n", g_pass, g_fail);
	return g_fail;
}
