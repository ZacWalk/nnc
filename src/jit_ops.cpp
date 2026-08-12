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
// gemv_bf16w_f32x(W, x, y)   — rows and cols baked.
//   rcx=W (bf16)  rdx=x (fp32)  r8=y (fp32)
//
// Internal register usage (low-8 only):
//   rcx -> advancing W row pointer (BF16, advances by cols*2 per row)
//   rdx -> x (constant)
//   rsi -> y base (saved/restored — was r8)
//   rdi -> row counter
//   rax -> inner column counter (advances by 8 or 32)
//   ymm0..ymm3 = accumulators (4 when cols%32==0, else just ymm0)
//   ymm4 = w (fp32), ymm5 = x load
//
// BF16 -> F32 is a free shift:
//   vpmovzxwd ymm, [m128]   ; 8 u16 -> 8 u32 (zero-ext)
//   vpslld    ymm, ymm, 16  ; bf16 bits -> high half of f32
//
// Stack: 2 push_r64 = 16 bytes => realigns to 16 (entry was 8 mod 16
// after the call). No further sub rsp needed; we make no calls.
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
//   r9 =scales (bf16, length cols/32, one per Q8_0 block)
//
// Writes *y_out = sum over blocks b of  scales[b] * sum_{k in b} qs[k] * x[k]
//
// Per-block inner-loop body (32 cols, 4 unrolled 8-col FMA steps):
//   vpbroadcastw y3, [rdi] ; vpslld y3,y3,16   ; bf16 block scale -> 8x f32
//   for k in 0,8,16,24:
//     vmovups   y1, [rdx + rax*4 + k*4]    ; x[rax+k..rax+k+7]
//     vpmovsxbd y2, [rcx + rax + k]        ; 8 i8 -> 8 i32
//     vcvtdq2ps y2, y2                     ; -> 8 f32
//     vmulps    y2, y2, y3                 ; * scale
//     vfmadd231ps y0, y2, y1               ; acc += scaled_q * x
//   add rax, 32 ; add rdi, 2 ; cmp rax, cols ; jl
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

	// Four independent accumulators (y0..y3): a single one would serialise
	// four 4-cycle FMAs per block and leave a thread latency-bound well
	// below what memory can feed. y4 = scale, y5 = weight scratch; x is a
	// memory operand on the FMA rather than a separate load.
	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
	v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
	v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);
	e.xor_r32_r32(gpr::rax, gpr::rax); // rax = col index

	const size_t block_loop = buf.size();

	v.vpbroadcastw_ymm_load_base(ymm::y4, gpr::rdi);
	v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);

	v.vpmovsxbd_ymm_load_basex1(ymm::y5, gpr::rcx, gpr::rax);
	v.vcvtdq2ps_ymm_ymm(ymm::y5, ymm::y5);
	v.vmulps_ymm_ymm_ymm(ymm::y5, ymm::y5, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex4(ymm::y0, ymm::y5, gpr::rdx, gpr::rax);

	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y5, gpr::rcx, gpr::rax, 8);
	v.vcvtdq2ps_ymm_ymm(ymm::y5, ymm::y5);
	v.vmulps_ymm_ymm_ymm(ymm::y5, ymm::y5, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex4_disp8(ymm::y1, ymm::y5, gpr::rdx, gpr::rax, 32);

	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y5, gpr::rcx, gpr::rax, 16);
	v.vcvtdq2ps_ymm_ymm(ymm::y5, ymm::y5);
	v.vmulps_ymm_ymm_ymm(ymm::y5, ymm::y5, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex4_disp8(ymm::y2, ymm::y5, gpr::rdx, gpr::rax, 64);

	v.vpmovsxbd_ymm_load_basex1_disp8(ymm::y5, gpr::rcx, gpr::rax, 24);
	v.vcvtdq2ps_ymm_ymm(ymm::y5, ymm::y5);
	v.vmulps_ymm_ymm_ymm(ymm::y5, ymm::y5, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex4_disp8(ymm::y3, ymm::y5, gpr::rdx, gpr::rax, 96);

	e.add_r64_imm8(gpr::rax, 32);
	e.add_r64_imm8(gpr::rdi, 2);
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols));
	emit_jl_back(buf, e, block_loop);

	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
	v.vaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y2);

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

// =====================================================================
// gemv_q4_s_1row(qs, x, y_out, scales)   — single-row 4-bit dot.
//   rcx=qs (packed nibbles, cols/2 bytes)
//   rdx=x  (fp32, cols floats)
//   r8 =y_out (one fp32 scalar)
//   r9 =scales (bf16, cols/32, one per block)
//
// Writes *y_out = sum over blocks b of  scales[b] * sum_{k in b} q[k]*x[k],
// with q the unsigned nibble in [0, 15]. The bias term is the caller's job.
//
// rax walks the qs byte offset (0, 16, 32, ...). Because 16 qs bytes cover
// 32 floats = 128 x bytes, the x address is exactly [rdx + rax*8] — one
// counter drives both streams.
//
// Per-block body (32 cols):
//   vpbroadcastw y3, [rdi] ; vpslld y3,y3,16  ; bf16 block scale -> 8x f32
//   vpmovzxbd    y2, [rcx + rax]        ; bytes 0..7   -> u32
//   vpmovzxbd    y6, [rcx + rax + 8]    ; bytes 8..15  -> u32
//   for (src, shift, xdisp) in (y2,and,0) (y6,and,32) (y2,shr,64) (y6,shr,96):
//     y5 = nibble(src) ; vcvtdq2ps ; vmulps y5,y5,y3
//     y1 = [rdx + rax*8 + xdisp]
//     vfmadd231ps y0, y5, y1
//   add rax,16 ; add rdi,2 ; cmp rax, cols/2 ; jl
//
// Registers: y0 acc, y1 x tile, y2/y6 raw quant bytes, y3 scale,
//            y4 0x0F mask (loop-invariant), y5 scratch.
// Stack: 2 push_r64 = 16 bytes. No additional sub rsp.
// =====================================================================
void nnc_build_gemv_q4_s_f32x_1row(jit_buffer& buf, const uint32_t cols)
{
	assert(cols > 0 && (cols % 32) == 0);

	x64_emitter e(buf);
	avx2_emitter v(buf);

	e.emit_win64_arg_shuffle(4); // (qs, x, y_out, scales)

	e.push_r64(gpr::rsi);
	e.push_r64(gpr::rdi);
	e.mov_r64_r64_srcext_ok(gpr::rsi, gpr::r8); // rsi = y_out
	e.mov_r64_r64_srcext_ok(gpr::rdi, gpr::r9); // rdi = scales

	// Four independent accumulators (y0..y3). One accumulator would chain
	// four 4-cycle FMAs per block and cap a thread at ~5 GB/s, well under
	// what the memory system can feed. y4 = scale, y5/y6 = the two loaded
	// byte groups, y7 = scratch. That leaves no register for a 0x0F mask,
	// so low nibbles are isolated with a shift pair instead, and x is read
	// straight into the FMA as a memory operand rather than via a load.
	v.vxorps_ymm(ymm::y0, ymm::y0, ymm::y0);
	v.vxorps_ymm(ymm::y1, ymm::y1, ymm::y1);
	v.vxorps_ymm(ymm::y2, ymm::y2, ymm::y2);
	v.vxorps_ymm(ymm::y3, ymm::y3, ymm::y3);
	e.xor_r32_r32(gpr::rax, gpr::rax); // rax = qs byte offset

	const size_t block_loop = buf.size();

	// bf16 scale -> f32 in all 8 lanes: broadcast the word, then shift it
	// into the high half of each dword.
	v.vpbroadcastw_ymm_load_base(ymm::y4, gpr::rdi);
	v.vpslld_ymm_imm8(ymm::y4, ymm::y4, 16);
	v.vpmovzxbd_ymm_load_basex1(ymm::y5, gpr::rcx, gpr::rax);
	v.vpmovzxbd_ymm_load_basex1_disp8(ymm::y6, gpr::rcx, gpr::rax, 8);

	// elements 0..7 : low nibbles of bytes 0..7
	v.vpslld_ymm_imm8(ymm::y7, ymm::y5, 28);
	v.vpsrld_ymm_imm8(ymm::y7, ymm::y7, 28);
	v.vcvtdq2ps_ymm_ymm(ymm::y7, ymm::y7);
	v.vmulps_ymm_ymm_ymm(ymm::y7, ymm::y7, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex8(ymm::y0, ymm::y7, gpr::rdx, gpr::rax);

	// elements 8..15 : low nibbles of bytes 8..15
	v.vpslld_ymm_imm8(ymm::y7, ymm::y6, 28);
	v.vpsrld_ymm_imm8(ymm::y7, ymm::y7, 28);
	v.vcvtdq2ps_ymm_ymm(ymm::y7, ymm::y7);
	v.vmulps_ymm_ymm_ymm(ymm::y7, ymm::y7, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex8_disp8(ymm::y1, ymm::y7, gpr::rdx, gpr::rax, 32);

	// elements 16..23 : high nibbles of bytes 0..7 (source was
	// zero-extended from a byte, so the shift needs no mask). y5 is dead
	// after this, so it doubles as the scratch register.
	v.vpsrld_ymm_imm8(ymm::y5, ymm::y5, 4);
	v.vcvtdq2ps_ymm_ymm(ymm::y5, ymm::y5);
	v.vmulps_ymm_ymm_ymm(ymm::y5, ymm::y5, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex8_disp8(ymm::y2, ymm::y5, gpr::rdx, gpr::rax, 64);

	// elements 24..31 : high nibbles of bytes 8..15
	v.vpsrld_ymm_imm8(ymm::y6, ymm::y6, 4);
	v.vcvtdq2ps_ymm_ymm(ymm::y6, ymm::y6);
	v.vmulps_ymm_ymm_ymm(ymm::y6, ymm::y6, ymm::y4);
	v.vfmadd231ps_ymm_mem_basex8_disp8(ymm::y3, ymm::y6, gpr::rdx, gpr::rax, 96);

	e.add_r64_imm8(gpr::rax, 16);
	e.add_r64_imm8(gpr::rdi, 2);
	e.cmp_r64_imm32(gpr::rax, static_cast<int32_t>(cols / 2));
	emit_jl_back(buf, e, block_loop);

	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y1);
	v.vaddps_ymm(ymm::y2, ymm::y2, ymm::y3);
	v.vaddps_ymm(ymm::y0, ymm::y0, ymm::y2);

	v.vextractf128_xmm_ymm(ymm::y1, ymm::y0, 1);
	v.vaddps_xmm(ymm::y0, ymm::y0, ymm::y1);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);
	v.vhaddps_xmm(ymm::y0, ymm::y0, ymm::y0);

	e.xor_r32_r32(gpr::rdi, gpr::rdi);
	v.vmovss_store_basex4(gpr::rsi, gpr::rdi, ymm::y0);

	v.vzeroupper();
	e.pop_r64(gpr::rdi);
	e.pop_r64(gpr::rsi);
	e.ret();
}
