// nnc — single test translation unit.
// All tests (app + jit) live here. Run with `nnc --test` (also -test, /test).
// Each test returns true on success. Exit code = number of failures.

#include "jit_buffer.h"
#include "jit_kernel.h"
#include "jit_ops.h"
#include "nn_ops.h"
#include "emitter_x64.h"

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
	report("nnc_gemv_f16w_f32x", test_nnc_gemv_f16w_f32x());

	std::printf("nnc: %d passed, %d failed\n", g_pass, g_fail);
	return g_fail;
}
