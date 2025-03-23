// nnc — high-level kernel builders.
// Each function here uses the x64 + AVX2 emitters to assemble a complete,
// shape-specialized kernel (dot_f32, gemv_f32, gemv_f16w_f32x, ...) into
// a jit_buffer ready for commit().

#include "jit_ops.h"

#include "emitter_avx2.h"
#include "jit_buffer.h"
#include "emitter_x64.h"

#include <cassert>
#include <cstdint>

// =====================================================================
// dot_f32(const float* a, const float* b, size_t n)
//   rcx=a  rdx=b  r8=n  -> xmm0
//   ymm0 = accumulator, ymm1 = load. rax = i.
// =====================================================================
void nnc_build_dot_f32(jit_buffer& buf)
{
	x64_emitter e(buf);
	avx2_emitter v(buf);

	e.xor_r32_r32(gpr::rax, gpr::rax);
	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);

	const size_t loop_start = buf.size();

	v.vmovups_ymm_load_basex4(ymm::y1, gpr::rcx, gpr::rax);
	v.vfmadd231ps_ymm_mem_basex4(ymm::y0, ymm::y1, gpr::rdx, gpr::rax);
	e.add_r64_imm8(gpr::rax, 8);
	e.cmp_r64_r64(gpr::rax, gpr::r8);
	const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 2;
	e.jl_rel8(static_cast<int8_t>(static_cast<ptrdiff_t>(loop_start) - after_jl));

	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	v.vzeroupper();
	e.ret();
}

// =====================================================================
// gemv_f32(W, x, y)   — rows and cols baked.
//   rcx=W  rdx=x  r8=y
//
// Internal register usage (low-8 only):
//   rcx -> advancing W row pointer
//   rdx -> x (constant)
//   rsi -> y base (saved/restored — was r8)
//   rdi -> row counter
//   rax -> inner column counter
//   ymm0 = accumulator, ymm1 = load
//
// Stack: 2 push_r64 = 16 bytes => realigns to 16 (entry was 8 mod 16
// after the call). No further sub rsp needed; we make no calls.
// =====================================================================
void nnc_build_gemv_f32(jit_buffer& buf, const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);
	// Inner-loop displacement uses rel8: distance must fit in -128.
	// At our current size (~12 bytes), this holds for all realistic cols.

	x64_emitter e(buf);
	avx2_emitter v(buf);

	// prologue
	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row = 0

	const int32_t col_bytes_per_row = static_cast<int32_t>(cols) * 4;

	// row_loop:
	const size_t row_loop = buf.size();

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	e.xor_r32_r32(gpr::rax, gpr::rax); // i = 0

	// col_loop:
	const size_t col_loop = buf.size();
	v.vmovups_ymm_load_basex4(ymm::y1, gpr::rcx, gpr::rax);
	v.vfmadd231ps_ymm_mem_basex4(ymm::y0, ymm::y1, gpr::rdx, gpr::rax);
	e.add_r64_imm8(gpr::rax, 8);
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	{
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 2;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(col_loop) - after_jl;
		assert(disp >= -128 && disp <= 127);
		e.jl_rel8(static_cast<int8_t>(disp));
	}

	// horizontal reduce ymm0 -> xmm0[0]
	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	// y[row] = xmm0[0]
	v.vmovss_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	// W += cols*4   ;   row += 1
	e.add_r64_imm32(gpr::rcx, col_bytes_per_row);
	e.add_r64_imm8(gpr::rdi, 1);
	e.cmp_r64_imm32(gpr::rdi, static_cast<int32_t>(rows));
	{
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 2;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(row_loop) - after_jl;
		assert(disp >= -128 && disp <= 127
			&& "row_loop too large for rel8; need rel32 jcc");
		e.jl_rel8(static_cast<int8_t>(disp));
	}

	// epilogue
	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}

// =====================================================================
// dot_f16_to_f32(const fp16* x, const fp16* y) -> float in xmm0
//   rcx=x  rdx=y
//
// Internal register usage (low-8 only, no save/restore):
//   ymm0..ymm3 = 4 accumulators
//   ymm4, ymm5 = working ax / ay
//
// Fully unrolled. n must be > 0 and a multiple of 32 (4*8 halves per iter).
// Each iteration consumes 64 bytes per input (32 halves x 2 bytes).
// =====================================================================
void nnc_build_dot_f16_to_f32(jit_buffer& buf, const uint32_t n)
{
	assert(n > 0);
	assert((n % 32) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
	v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
	v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);

	const uint32_t iters = n / 32;
	for (uint32_t i = 0; i < iters; ++i)
	{
		const int32_t base = static_cast<int32_t>(i) * 64;

		v.vcvtph2ps_ymm_load_base_disp32(ymm::y4, gpr::rcx, base + 0);
		v.vcvtph2ps_ymm_load_base_disp32(ymm::y5, gpr::rdx, base + 0);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_base_disp32(ymm::y4, gpr::rcx, base + 16);
		v.vcvtph2ps_ymm_load_base_disp32(ymm::y5, gpr::rdx, base + 16);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y1, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_base_disp32(ymm::y4, gpr::rcx, base + 32);
		v.vcvtph2ps_ymm_load_base_disp32(ymm::y5, gpr::rdx, base + 32);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y2, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_base_disp32(ymm::y4, gpr::rcx, base + 48);
		v.vcvtph2ps_ymm_load_base_disp32(ymm::y5, gpr::rdx, base + 48);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y3, ymm::y4, ymm::y5);
	}

	// reduce ymm0..ymm3 -> ymm0
	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
	v.vaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y2);

	// horizontal reduce ymm0 -> xmm0[0]
	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	v.vzeroupper();
	e.ret();
}

// =====================================================================
// gemv_f16w_f32x(W, x, y)   — rows and cols baked.
//   rcx=W (fp16)  rdx=x (fp32)  r8=y (fp32)
//
// Internal register usage (low-8 only):
//   rcx -> advancing W row pointer (FP16, advances by cols*2 per row)
//   rdx -> x (constant)
//   rsi -> y base (saved/restored — was r8)
//   rdi -> row counter
//   rax -> inner column counter (advances by 8 or 32)
//   ymm0..ymm3 = accumulators (4 when cols%32==0, else just ymm0)
//   ymm4 = w (fp32 from vcvtph2ps), ymm5 = x load
//
// Stack: 2 push_r64 = 16 bytes => realigns to 16 (entry was 8 mod 16
// after the call). No further sub rsp needed; we make no calls.
// =====================================================================
void nnc_build_gemv_f16w_f32x(jit_buffer& buf, const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	// Use 4 accumulators with stride 32 when cols is a multiple of 32.
	// This breaks the FMA dependency chain (4-cycle latency -> 1-cycle
	// throughput) and roughly quadruples the inner-loop ILP.
	const bool unroll4 = (cols % 32) == 0;

	// prologue
	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row = 0

	const int32_t row_bytes = static_cast<int32_t>(cols) * 2; // FP16 row stride

	// row_loop:
	const size_t row_loop = buf.size();

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	if (unroll4)
	{
		v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
		v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
		v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);
	}
	e.xor_r32_r32(gpr::rax, gpr::rax); // i = 0

	// col_loop:
	const size_t col_loop = buf.size();
	if (unroll4)
	{
		// 4 independent FMA chains, 32 columns per iteration.
		// w base = rcx + rax*2 + {0,16,32,48} ; x base = rdx + rax*4 + {0,32,64,96}
		v.vcvtph2ps_ymm_load_basex2(ymm::y4, gpr::rcx, gpr::rax);
		v.vmovups_ymm_load_basex4(ymm::y5, gpr::rdx, gpr::rax);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 16);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 32);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y1, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 32);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 64);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y2, ymm::y4, ymm::y5);

		v.vcvtph2ps_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 48);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 96);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y3, ymm::y4, ymm::y5);

		e.add_r64_imm8(gpr::rax, 32);
	}
	else
	{
		v.vcvtph2ps_ymm_load_basex2(ymm::y4, gpr::rcx, gpr::rax);
		v.vmovups_ymm_load_basex4(ymm::y5, gpr::rdx, gpr::rax);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y4, ymm::y5);
		e.add_r64_imm8(gpr::rax, 8);
	}
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	{
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 2;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(col_loop) - after_jl;
		assert(disp >= -128 && disp <= 127);
		e.jl_rel8(static_cast<int8_t>(disp));
	}

	// reduce 4 accumulators -> ymm0 (only needed for unroll4 path)
	if (unroll4)
	{
		v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
		v.vaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
		v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y2);
	}

	// horizontal reduce ymm0 -> xmm0[0]
	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	// y[row] = xmm0[0]
	v.vmovss_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	// W += cols*2   ;   row += 1
	e.add_r64_imm32(gpr::rcx, row_bytes);
	e.add_r64_imm8(gpr::rdi, 1);
	e.cmp_r64_imm32(gpr::rdi, static_cast<int32_t>(rows));
	{
		// Use rel32 unconditionally — the row body is large and grows
		// with the unrolled inner loop; rel8 is too tight to rely on.
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 6;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(row_loop) - after_jl;
		e.jl_rel32(static_cast<int32_t>(disp));
	}

	// epilogue
	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}
