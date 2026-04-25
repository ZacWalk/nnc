// nnc — AVX2 + FMA encoders.
// VEX-encoded instructions only. All public ops here use low-8 ymm/xmm
// registers and low-8 GPR base/index. Extended-register support will be
// added when a kernel actually needs it.

#pragma once

#include <cstdint>

#include "emitter_x64.h"   // for gpr enum

class jit_buffer;

enum class ymm : uint8_t { y0 = 0, y1, y2, y3, y4, y5, y6, y7 };

class avx2_emitter
{
public:
	explicit avx2_emitter(jit_buffer& buf) : buf_(buf)
	{
	}

	// ---- raw VEX builders ----------------------------------------------

	void vex2(uint8_t r, uint8_t vvvv, uint8_t l, uint8_t pp);
	void vex3(uint8_t r, uint8_t x, uint8_t b,
	          uint8_t map, uint8_t w, uint8_t vvvv, uint8_t l, uint8_t pp);

	// ---- arithmetic / data movement ------------------------------------

	void vxorps_ymm(ymm dst, ymm a, ymm b);

	// ymm dst <- [base + index*4]
	void vmovups_ymm_load_basex4(ymm dst, gpr base, gpr index);

	// ymm dst <- [base + index*4 + disp8]   (mod=01 SIB form)
	void vmovups_ymm_load_basex4_disp8(ymm dst, gpr base, gpr index, int8_t disp);

	// acc <- acc + a * [base + index*4]
	void vfmadd231ps_ymm_mem_basex4(ymm acc, ymm a, gpr base, gpr index);

	// acc <- acc + a * b   (register-register form)
	void vfmadd231ps_ymm_ymm_ymm(ymm acc, ymm a, ymm b);

	// ymm dst <- vcvtph2ps(xmmword [base + disp32])  (8 halves -> 8 floats)
	void vcvtph2ps_ymm_load_base_disp32(ymm dst, gpr base, int32_t disp);

	// ymm dst <- vcvtph2ps(xmmword [base + index*2])  (8 halves -> 8 floats)
	void vcvtph2ps_ymm_load_basex2(ymm dst, gpr base, gpr index);

	// ymm dst <- vcvtph2ps(xmmword [base + index*2 + disp8])  (mod=01 SIB)
	void vcvtph2ps_ymm_load_basex2_disp8(ymm dst, gpr base, gpr index, int8_t disp);

	// ymm dst <- vpmovzxwd(xmmword [base + index*2])  (8 u16 -> 8 u32, zero-ext)
	void vpmovzxwd_ymm_load_basex2(ymm dst, gpr base, gpr index);

	// ymm dst <- vpmovzxwd(xmmword [base + index*2 + disp8])  (mod=01 SIB)
	void vpmovzxwd_ymm_load_basex2_disp8(ymm dst, gpr base, gpr index, int8_t disp);

	// ymm dst <- vpmovzxwd(xmmword [base + index*2 + disp32])  (mod=10 SIB)
	void vpmovzxwd_ymm_load_basex2_disp32(ymm dst, gpr base, gpr index, int32_t disp);

	// ymm dst <- vpslld(ymm src, imm8)   (logical shift left, 32-bit lanes)
	void vpslld_ymm_imm8(ymm dst, ymm src, uint8_t imm);

	// xmm dst <- ymm src lane (0 = low, 1 = high)
	void vextractf128_xmm_ymm(ymm dst_xmm, ymm src_ymm, uint8_t lane);

	void vaddps_xmm(ymm dst, ymm a, ymm b);
	void vaddps_ymm(ymm dst, ymm a, ymm b);
	void vhaddps_xmm(ymm dst, ymm a, ymm b);
	void vhaddps_ymm(ymm dst, ymm a, ymm b);

	// [base + index*4]  <-  xmm src lane 0  (32-bit store)
	void vmovss_store_basex4(gpr base, gpr index, ymm src_xmm);

	// [base + index*4]  <-  xmm src (128-bit / 16-byte unaligned store)
	void vmovups_xmm_store_basex4(gpr base, gpr index, ymm src_xmm);

	void vzeroupper();

private:
	jit_buffer& buf_;
};
