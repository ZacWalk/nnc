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

// Emit a JL targeting absolute buffer offset `target` from the current
// buf.size(). Picks rel8 when the displacement fits, falls back to rel32
// otherwise so kernels keep building if their bodies grow.
static void emit_jl_back(const jit_buffer& buf, x64_emitter& e, const size_t target)
{
	const ptrdiff_t after_rel8 = static_cast<ptrdiff_t>(buf.size()) + 2;
	const ptrdiff_t disp8 = static_cast<ptrdiff_t>(target) - after_rel8;
	if (disp8 >= -128 && disp8 <= 127)
	{
		e.jl_rel8(static_cast<int8_t>(disp8));
	}
	else
	{
		const ptrdiff_t after_rel32 = static_cast<ptrdiff_t>(buf.size()) + 6;
		const ptrdiff_t disp32 = static_cast<ptrdiff_t>(target) - after_rel32;
		e.jl_rel32(static_cast<int32_t>(disp32));
	}
}

// =====================================================================
// dot_f32(const float* a, const float* b, size_t n)
//   rcx=a  rdx=b  r8=n  -> xmm0
//   ymm0 = accumulator, ymm1 = load. rax = i.
// =====================================================================
void nnc_build_dot_f32(jit_buffer& buf)
{
	x64_emitter e(buf);
	avx2_emitter v(buf);

	e.emit_win64_arg_shuffle(3); // (a, b, n)

	e.xor_r32_r32(gpr::rax, gpr::rax);
	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);

	const size_t loop_start = buf.size();

	v.vmovups_ymm_load_basex4(ymm::y1, gpr::rcx, gpr::rax);
	v.vfmadd231ps_ymm_mem_basex4(ymm::y0, ymm::y1, gpr::rdx, gpr::rax);
	e.add_r64_imm8(gpr::rax, 8);
	e.cmp_r64_r64(gpr::rax, gpr::r8);
	emit_jl_back(buf, e, loop_start);

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

	e.emit_win64_arg_shuffle(3); // (W, x, y)

	// prologue
	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row = 0

	const int64_t cbpr64 = static_cast<int64_t>(cols) * 4;
	assert(cbpr64 <= INT32_MAX && "gemv_f32: cols too large for int32 row stride");
	const int32_t col_bytes_per_row = static_cast<int32_t>(cbpr64);

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
	emit_jl_back(buf, e, col_loop);

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
	emit_jl_back(buf, e, row_loop);

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

	e.emit_win64_arg_shuffle(2); // (x, y)

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

	e.emit_win64_arg_shuffle(3); // (W, x, y)

	// prologue
	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row = 0

	const int64_t row_bytes64 = static_cast<int64_t>(cols) * 2; // FP16 row stride
	assert(row_bytes64 <= INT32_MAX && "gemv_f16w: cols too large for int32 row stride");
	const int32_t row_bytes = static_cast<int32_t>(row_bytes64);

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
	emit_jl_back(buf, e, col_loop);

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

// =====================================================================
// gemv_bf16w_f32x(W, x, y)   — rows and cols baked.
//   rcx=W (bf16)  rdx=x (fp32)  r8=y (fp32)
//
// Identical structure to gemv_f16w_f32x, but BF16 -> F32 uses
//   vpmovzxwd ymm, [m128]   ; 8 u16 -> 8 u32 (zero-ext)
//   vpslld    ymm, ymm, 16  ; bf16 bits -> high half of f32
// instead of vcvtph2ps. BF16 row stride is also cols*2 bytes.
// =====================================================================
void nnc_build_gemv_bf16w_f32x(jit_buffer& buf, const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0);
	assert(cols > 0 && (cols % 8) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	const bool unroll4 = (cols % 32) == 0;

	e.emit_win64_arg_shuffle(3); // (W, x, y)

	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row = 0

	const int64_t row_bytes64 = static_cast<int64_t>(cols) * 2; // BF16 row stride
	assert(row_bytes64 <= INT32_MAX && "gemv_bf16w: cols too large for int32 row stride");
	const int32_t row_bytes = static_cast<int32_t>(row_bytes64);

	const size_t row_loop = buf.size();

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	if (unroll4)
	{
		v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
		v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
		v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);
	}
	e.xor_r32_r32(gpr::rax, gpr::rax); // i = 0

	const size_t col_loop = buf.size();
	if (unroll4)
	{
		v.vpmovzxwd_ymm_load_basex2(ymm::y4, gpr::rcx, gpr::rax);
		v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
		v.vmovups_ymm_load_basex4(ymm::y5, gpr::rdx, gpr::rax);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y4, ymm::y5);

		v.vpmovzxwd_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 16);
		v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 32);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y1, ymm::y4, ymm::y5);

		v.vpmovzxwd_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 32);
		v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 64);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y2, ymm::y4, ymm::y5);

		v.vpmovzxwd_ymm_load_basex2_disp8(ymm::y4, gpr::rcx, gpr::rax, 48);
		v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
		v.vmovups_ymm_load_basex4_disp8(ymm::y5, gpr::rdx, gpr::rax, 96);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y3, ymm::y4, ymm::y5);

		e.add_r64_imm8(gpr::rax, 32);
	}
	else
	{
		v.vpmovzxwd_ymm_load_basex2(ymm::y4, gpr::rcx, gpr::rax);
		v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
		v.vmovups_ymm_load_basex4(ymm::y5, gpr::rdx, gpr::rax);
		v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y4, ymm::y5);
		e.add_r64_imm8(gpr::rax, 8);
	}
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	emit_jl_back(buf, e, col_loop);

	if (unroll4)
	{
		v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
		v.vaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
		v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y2);
	}

	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	v.vmovss_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	e.add_r64_imm32(gpr::rcx, row_bytes);
	e.add_r64_imm8(gpr::rdi, 1);
	e.cmp_r64_imm32(gpr::rdi, static_cast<int32_t>(rows));
	{
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 6;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(row_loop) - after_jl;
		e.jl_rel32(static_cast<int32_t>(disp));
	}

	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}

// =====================================================================
// gemv_bf16w_f32x_4row(W, x, y)   — rows and cols baked. 4 rows / iter.
//   rcx=W (bf16)  rdx=x (fp32)  r8=y (fp32)
//
// Inner-loop body (one 8-col tile per iter):
//   ymm4   = vmovups   [rdx + rax*4]                ; x[k..k+7]
//   ymm5   = vpmovzxwd [rcx + rax*2 + 0*RB] ; vpslld 16 ; vfmadd y0,y5,y4
//   ymm5   = vpmovzxwd [rcx + rax*2 + 1*RB] ; vpslld 16 ; vfmadd y1,y5,y4
//   ymm5   = vpmovzxwd [rcx + rax*2 + 2*RB] ; vpslld 16 ; vfmadd y2,y5,y4
//   ymm5   = vpmovzxwd [rcx + rax*2 + 3*RB] ; vpslld 16 ; vfmadd y3,y5,y4
//   add rax, 8 ; cmp rax, cols ; jl
//
// Tail reduction (4 ymms -> 4 contiguous floats):
//   y0 = vhaddps_ymm(y0, y1)       ; per-lane: (a01,a23,b01,b23) x 2
//   y2 = vhaddps_ymm(y2, y3)
//   y0 = vhaddps_ymm(y0, y2)       ; per-lane: (a,b,c,d) x 2
//   x1 = vextractf128 y0, 1
//   x0 = vaddps_xmm(x0, x1)        ; [sum_a, sum_b, sum_c, sum_d]
//   vmovups [rsi + rdi*4], xmm0    ; 16-byte store
//
// Then: add rcx, 4*RB ; add rdi, 4 ; cmp rdi, rows ; jl
//
// Stack: 2 push_r64 = 16 bytes. No additional sub rsp.
// =====================================================================
void nnc_build_gemv_bf16w_f32x_4row(jit_buffer& buf, const uint32_t rows, const uint32_t cols)
{
	assert(rows > 0 && (rows % 4) == 0);
	assert(cols > 0 && (cols % 8) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	const int64_t row_bytes64 = static_cast<int64_t>(cols) * 2; // BF16 row stride
	assert(row_bytes64 <= INT32_MAX && "gemv_bf16w: cols too large for int32 row stride");
	const int32_t row_bytes = static_cast<int32_t>(row_bytes64);

	e.emit_win64_arg_shuffle(3); // (W, x, y)

	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y
	e.xor_r32_r32(gpr::rdi, gpr::rdi); // row_group = 0

	const size_t row_loop = buf.size();

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
	v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
	v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);
	e.xor_r32_r32(gpr::rax, gpr::rax); // i = 0

	const size_t col_loop = buf.size();

	// x tile (shared across all 4 row FMAs)
	v.vmovups_ymm_load_basex4(ymm::y4, gpr::rdx, gpr::rax);

	// Row 0: offset 0 from rcx (no displacement form).
	v.vpmovzxwd_ymm_load_basex2(ymm::y5, gpr::rcx, gpr::rax);
	v.vpslld_ymm_imm8(ymm::y5, ymm::y5, 16);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y5, ymm::y4);

	// Rows 1..3: disp32 (row_bytes can be > 127, e.g. 4096 for cols=2048).
	v.vpmovzxwd_ymm_load_basex2_disp32(ymm::y5, gpr::rcx, gpr::rax, row_bytes);
	v.vpslld_ymm_imm8(ymm::y5, ymm::y5, 16);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y1, ymm::y5, ymm::y4);

	v.vpmovzxwd_ymm_load_basex2_disp32(ymm::y5, gpr::rcx, gpr::rax, row_bytes * 2);
	v.vpslld_ymm_imm8(ymm::y5, ymm::y5, 16);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y2, ymm::y5, ymm::y4);

	v.vpmovzxwd_ymm_load_basex2_disp32(ymm::y5, gpr::rcx, gpr::rax, row_bytes * 3);
	v.vpslld_ymm_imm8(ymm::y5, ymm::y5, 16);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y3, ymm::y5, ymm::y4);

	e.add_r64_imm8(gpr::rax, 8);
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	emit_jl_back(buf, e, col_loop);

	// Reduce 4 ymm partial sums into 4 contiguous floats in xmm0.
	v.vhaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vhaddps_ymm(ymm::y0, ymm::y0, ymm::y2);
	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);

	// Store 4 floats at y[rdi .. rdi+3].
	v.vmovups_xmm_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	// Advance to next row-group of 4.
	e.add_r64_imm32(gpr::rcx, row_bytes * 4);
	e.add_r64_imm8(gpr::rdi, 4);
	e.cmp_r64_imm32(gpr::rdi, static_cast<int32_t>(rows));
	{
		const ptrdiff_t after_jl = static_cast<ptrdiff_t>(buf.size()) + 6;
		const ptrdiff_t disp = static_cast<ptrdiff_t>(row_loop) - after_jl;
		e.jl_rel32(static_cast<int32_t>(disp));
	}

	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}

// =====================================================================
// gemv_q8_0_f32x_1row(qs, x, y_out, scales)   — single-row Q8_0 dot.
//   rcx=qs (int8 row, length cols)
//   rdx=x  (fp32, length cols)
//   r8 =y_out (one fp32 scalar)
//   r9 =scales (fp32, length cols/32, one per Q8_0 block)
//
// Writes *y_out = sum over blocks b of  scales[b] * sum_{k in b} qs[k] * x[k]
//
// Per-block inner-loop body (32 cols, 4 unrolled 8-col FMA steps):
//   vbroadcastss y3, [rdi]         ; broadcast block scale
//   for k in 0,8,16,24:
//     vmovups   y1, [rdx + rax*4 + k*4]    ; x[rax+k..rax+k+7]
//     vpmovsxbd y2, [rcx + rax + k]        ; 8 i8 -> 8 i32
//     vcvtdq2ps y2, y2                     ; -> 8 f32
//     vmulps    y2, y2, y3                 ; * scale
//     vfmadd231ps y0, y2, y1               ; acc += scaled_q * x
//   add rax, 32 ; add rdi, 4 ; cmp rax, cols ; jl
//
// We pre-multiply qs by the scale and FMA with x (rather than FMA qs*x
// then mul-add scale at the end of the block) because that keeps a
// single ymm accumulator and avoids needing 4 partial sums per block.
// The extra vmulps per inner step is negligible — decode is BW bound.
// =====================================================================
void nnc_build_gemv_q8_0_f32x_1row(jit_buffer& buf, const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	e.emit_win64_arg_shuffle(4); // (qs, x, y_out, scales)

	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y_out
	e.mov_r64_r64_srcext_ok(gpr::rdi, gpr::r9); // rdi = scales

	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	e.xor_r32_r32(gpr::rax, gpr::rax); // rax = col index

	const size_t block_loop = buf.size();

	v.vbroadcastss_ymm_load_base(ymm::y3, gpr::rdi);

	v.vmovups_ymm_load_basex4(ymm::y1, gpr::rdx, gpr::rax);
	v.vpmovsxbd_ymm_load_basex1(ymm::y2, gpr::rcx, gpr::rax);
	v.vcvtdq2ps_ymm_ymm(ymm::y2, ymm::y2);
	v.vmulps_ymm_ymm_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y2, ymm::y1);

	v.vmovups_ymm_load_basex4_disp8(ymm::y1, gpr::rdx, gpr::rax, 32);
	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y2, gpr::rcx, gpr::rax, 8);
	v.vcvtdq2ps_ymm_ymm(ymm::y2, ymm::y2);
	v.vmulps_ymm_ymm_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y2, ymm::y1);

	v.vmovups_ymm_load_basex4_disp8(ymm::y1, gpr::rdx, gpr::rax, 64);
	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y2, gpr::rcx, gpr::rax, 16);
	v.vcvtdq2ps_ymm_ymm(ymm::y2, ymm::y2);
	v.vmulps_ymm_ymm_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y2, ymm::y1);

	v.vmovups_ymm_load_basex4_disp8(ymm::y1, gpr::rdx, gpr::rax, 96);
	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y2, gpr::rcx, gpr::rax, 24);
	v.vcvtdq2ps_ymm_ymm(ymm::y2, ymm::y2);
	v.vmulps_ymm_ymm_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vfmadd231ps_ymm_ymm_ymm(ymm::y0, ymm::y2, ymm::y1);

	e.add_r64_imm8(gpr::rax, 32);
	e.add_r64_imm8(gpr::rdi, 4);
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	emit_jl_back(buf, e, block_loop);

	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	// rdi no longer needed for scales — reuse as zero index for the
	// existing SIB-form vmovss store: vmovss [rsi + rdi*4], xmm0.
	e.xor_r32_r32(gpr::rdi, gpr::rdi);
	v.vmovss_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}
