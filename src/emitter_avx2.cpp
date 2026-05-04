// nnc — AVX2 + FMA + F16C encoder implementation.
// VEX 2-byte / 3-byte prefix construction plus the specific vector
// instructions used by the JIT kernels (loads, fmadd, broadcast,
// vcvtph2ps, horizontal helpers, ...).

#include "emitter_avx2.h"
#include "jit_buffer.h"

#include <cassert>

static inline uint8_t modrm(const uint8_t mod, const uint8_t reg, const uint8_t rm)
{
	return static_cast<uint8_t>((mod << 6) | ((reg & 7) << 3) | (rm & 7));
}

static inline uint8_t sib(const uint8_t scale, const uint8_t index, const uint8_t base)
{
	return static_cast<uint8_t>((scale << 6) | ((index & 7) << 3) | (base & 7));
}

// VEX bit fields are stored *inverted* relative to REX.
static inline uint8_t vex_inv_high(const uint8_t r) { return static_cast<uint8_t>(((r >> 3) & 1) ^ 1); }
static inline uint8_t vex_inv_vvvv(const uint8_t r) { return static_cast<uint8_t>((~r) & 0xF); }

void avx2_emitter::vex2(const uint8_t r, const uint8_t vvvv, const uint8_t l, const uint8_t pp)
{
	const uint8_t b1 = static_cast<uint8_t>(((r & 1) << 7) | ((vvvv & 0xF) << 3) | ((l & 1) << 2) | (pp & 3));
	buf_.emit_u8(0xC5);
	buf_.emit_u8(b1);
}

void avx2_emitter::vex3(const uint8_t r, const uint8_t x, const uint8_t b,
                        const uint8_t map, const uint8_t w, const uint8_t vvvv, const uint8_t l, const uint8_t pp)
{
	const uint8_t b1 = static_cast<uint8_t>(((r & 1) << 7) | ((x & 1) << 6) | ((b & 1) << 5) | (map & 0x1F));
	const uint8_t b2 = static_cast<uint8_t>(((w & 1) << 7) | ((vvvv & 0xF) << 3) | ((l & 1) << 2) | (pp & 3));
	buf_.emit_u8(0xC4);
	buf_.emit_u8(b1);
	buf_.emit_u8(b2);
}

// ----- VEX.NDS.256.0F.WIG 57 /r : VXORPS ymm, ymm, ymm -------------------
void avx2_emitter::vxorps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(b);
	assert(d < 8 && static_cast<uint8_t>(a) < 8 && s < 8);
	vex2(vex_inv_high(d), vex_inv_vvvv(static_cast<uint8_t>(a)), 1, 0);
	buf_.emit_u8(0x57);
	buf_.emit_u8(modrm(0b11, d, s));
}

// ----- VEX.256.0F.WIG 10 /r : VMOVUPS ymm, m256 --------------------------
void avx2_emitter::vmovups_ymm_load_basex4(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VMOVUPS ymm, [base + index*4 + disp8] -----------------------------
void avx2_emitter::vmovups_ymm_load_basex4_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, m256 ---------
void avx2_emitter::vfmadd231ps_ymm_mem_basex4(ymm acc, ymm a, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, [base+ix*4+disp8]
void avx2_emitter::vfmadd231ps_ymm_mem_basex4_disp8(ymm acc, ymm a, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, ymm ---------
void avx2_emitter::vfmadd231ps_ymm_ymm_ymm(ymm acc, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex3(vex_inv_high(d), 1, vex_inv_high(s2),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.256.66.0F38.W0 13 /r : VCVTPH2PS ymm, m128 -------------------
void avx2_emitter::vcvtph2ps_ymm_load_base_disp32(ymm dst, gpr base, const int32_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	assert(d < 8 && bs < 8);
	assert((bs & 7) != 4 && "base = RSP/R12 needs SIB; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x13);
	// mod=10 disp32 form: works for any low-8 base except rsp (needs SIB)
	// and is also fine for rbp (mod=00 would be RIP-relative, not what we want).
	buf_.emit_u8(modrm(0b10, d, bs));
	buf_.emit_u32(static_cast<uint32_t>(disp));
}

// ----- VEX.256.66.0F38.W0 13 /r : VCVTPH2PS ymm, m128 (SIB index*2) ------
void avx2_emitter::vcvtph2ps_ymm_load_basex2(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x13);
	buf_.emit_u8(modrm(0b00, d, 0b100)); // SIB follows
	buf_.emit_u8(sib(0b01, ix, bs)); // scale=*2
}

// ----- VCVTPH2PS ymm, [base + index*2 + disp8] --------------------------
void avx2_emitter::vcvtph2ps_ymm_load_basex2_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x13);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b01, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.256.66.0F38.W0 33 /r : VPMOVZXWD ymm, m128 (SIB index*2) -----
void avx2_emitter::vpmovzxwd_ymm_load_basex2(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b00, d, 0b100)); // SIB follows
	buf_.emit_u8(sib(0b01, ix, bs)); // scale=*2
}

// ----- VPMOVZXWD ymm, [base + index*2 + disp8] --------------------------
void avx2_emitter::vpmovzxwd_ymm_load_basex2_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b01, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VPMOVZXWD ymm, [base + index*2 + disp32] -------------------------
void avx2_emitter::vpmovzxwd_ymm_load_basex2_disp32(ymm dst, gpr base, gpr index, const int32_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b10, d, 0b100));
	buf_.emit_u8(sib(0b01, ix, bs));
	buf_.emit_u32(static_cast<uint32_t>(disp));
}

// ----- VEX.256.66.0F38.W0 21 /r : VPMOVSXBD ymm, m64 (SIB index*1) -------
// 8 packed signed bytes in [base + index] -> 8 sign-extended dwords in ymm.
void avx2_emitter::vpmovsxbd_ymm_load_basex1(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x21);
	buf_.emit_u8(modrm(0b00, d, 0b100)); // SIB follows
	buf_.emit_u8(sib(0b00, ix, bs)); // scale = *1
}

// ----- VPMOVSXBD ymm, [base + index + disp8] -----------------------------
void avx2_emitter::vpmovsxbd_ymm_load_basex1_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x21);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b00, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.256.66.0F38.W0 18 /r : VBROADCASTSS ymm, m32 -----------------
// Loads one f32 from [base] and broadcasts to all 8 lanes.
void avx2_emitter::vbroadcastss_ymm_load_base(ymm dst, gpr base)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	assert(d < 8 && bs < 8);
	assert((bs & 7) != 4 && "base = RSP/R12 needs SIB; not implemented");
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x18);
	buf_.emit_u8(modrm(0b00, d, bs));
}

// ----- VEX.256.0F.WIG 5B /r : VCVTDQ2PS ymm, ymm ------------------------
// 8 packed signed dwords -> 8 packed single-precision floats.
void avx2_emitter::vcvtdq2ps_ymm_ymm(ymm dst, ymm src)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);

	vex2(vex_inv_high(d), 0xF, 1, 0); // L=1 (256), pp=none
	buf_.emit_u8(0x5B);
	buf_.emit_u8(modrm(0b11, d, s));
}

// ----- VEX.NDS.256.0F.WIG 59 /r : VMULPS ymm, ymm, ymm ------------------
void avx2_emitter::vmulps_ymm_ymm_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0); // L=1, pp=none
	buf_.emit_u8(0x59);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.66.0F.WIG 72 /6 ib : VPSLLD ymm, ymm, imm8 -----------
// dst encoded in vvvv, src in modrm.rm, /6 in modrm.reg.
void avx2_emitter::vpslld_ymm_imm8(ymm dst, ymm src, const uint8_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);

	vex2(1, vex_inv_vvvv(d), 1, 0b01); // L=1 (256), pp=66
	buf_.emit_u8(0x72);
	buf_.emit_u8(modrm(0b11, 6, s));
	buf_.emit_u8(imm);
}

// ----- VEX.256.66.0F3A.W0 19 /r ib : VEXTRACTF128 xmm, ymm, imm8 ---------
void avx2_emitter::vextractf128_xmm_ymm(ymm dst_xmm, ymm src_ymm, const uint8_t lane)
{
	const uint8_t d = static_cast<uint8_t>(dst_xmm);
	const uint8_t s = static_cast<uint8_t>(src_ymm);
	assert(d < 8 && s < 8);
	assert(lane < 2);

	vex3(vex_inv_high(s), 1, vex_inv_high(d),
	     0b00011, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x19);
	buf_.emit_u8(modrm(0b11, s, d));
	buf_.emit_u8(lane);
}

// ----- VEX.NDS.128.0F.WIG 58 /r : VADDPS xmm, xmm, xmm -------------------
void avx2_emitter::vaddps_xmm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 0, 0);
	buf_.emit_u8(0x58);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.0F.WIG 58 /r : VADDPS ymm, ymm, ymm -------------------
void avx2_emitter::vaddps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0);
	buf_.emit_u8(0x58);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.128.F2.0F.WIG 7C /r : VHADDPS xmm, xmm, xmm ---------------
void avx2_emitter::vhaddps_xmm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 0, 0b11);
	buf_.emit_u8(0x7C);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.F2.0F.WIG 7C /r : VHADDPS ymm, ymm, ymm ---------------
// 256-bit form: hadd is performed per 128-bit lane (does NOT cross lanes).
void avx2_emitter::vhaddps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0b11);
	buf_.emit_u8(0x7C);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.128.0F.WIG 11 /r : VMOVUPS m128, xmm --------------------------
void avx2_emitter::vmovups_xmm_store_basex4(gpr base, gpr index, ymm src_xmm)
{
	const uint8_t s = static_cast<uint8_t>(src_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(s < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex2(vex_inv_high(s), 0xF, 0, 0); // pp = none, L=0
	buf_.emit_u8(0x11);
	buf_.emit_u8(modrm(0b00, s, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VEX.128.F3.0F.WIG 11 /r : VMOVSS m32, xmm -------------------------
void avx2_emitter::vmovss_store_basex4(gpr base, gpr index, ymm src_xmm)
{
	const uint8_t s = static_cast<uint8_t>(src_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(s < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex2(vex_inv_high(s), 0xF, 0, 0b10); // pp = F3
	buf_.emit_u8(0x11);
	buf_.emit_u8(modrm(0b00, s, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VEX.128.0F.WIG 77 : VZEROUPPER -----------------------------------
void avx2_emitter::vzeroupper()
{
	vex2(1, 0xF, 0, 0);
	buf_.emit_u8(0x77);
}
