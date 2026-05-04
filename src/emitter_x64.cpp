// nnc — x86-64 instruction encoder implementation.
// REX / ModR/M / SIB / displacement / immediate byte emission for the
// subset of integer/scalar instructions nnc actually uses.

#include "emitter_x64.h"
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

void x64_emitter::mov_r32_imm32(gpr dst, const uint32_t imm)
{
	const uint8_t r = static_cast<uint8_t>(dst);
	assert(r < 8);
	buf_.emit_u8(static_cast<uint8_t>(0xB8 + (r & 7)));
	buf_.emit_u32(imm);
}

void x64_emitter::xor_r32_r32(gpr dst, gpr src)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);
	// 31 /r : xor r/m32, r32. ModRM: reg=src, rm=dst.
	buf_.emit_u8(0x31);
	buf_.emit_u8(modrm(0b11, s, d));
}

void x64_emitter::mov_r64_r64_srcext_ok(gpr dst_low8, gpr src_any)
{
	const uint8_t d = static_cast<uint8_t>(dst_low8);
	const uint8_t s = static_cast<uint8_t>(src_any);
	assert(d < 8);
	// REX.W=1, REX.R=src high bit, REX.B=dst high bit (=0 here).
	const uint8_t rex = static_cast<uint8_t>(0x48 | (((s >> 3) & 1) << 2));
	buf_.emit_u8(rex);
	buf_.emit_u8(0x89); // mov r/m64, r64
	buf_.emit_u8(modrm(0b11, s & 7, d));
}

void x64_emitter::add_r64_imm8(gpr dst, const int8_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	assert(d < 8);
	buf_.emit_u8(0x48); // REX.W
	buf_.emit_u8(0x83); // 83 /0 ib
	buf_.emit_u8(modrm(0b11, 0, d));
	buf_.emit_u8(static_cast<uint8_t>(imm));
}

void x64_emitter::add_r64_imm32(gpr dst, const int32_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	assert(d < 8);
	buf_.emit_u8(0x48); // REX.W
	buf_.emit_u8(0x81); // 81 /0 id
	buf_.emit_u8(modrm(0b11, 0, d));
	buf_.emit_u32(static_cast<uint32_t>(imm));
}

void x64_emitter::cmp_r64_imm32(gpr dst, const int32_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	assert(d < 8);
	buf_.emit_u8(0x48); // REX.W
	buf_.emit_u8(0x81); // 81 /7 id
	buf_.emit_u8(modrm(0b11, 7, d));
	buf_.emit_u32(static_cast<uint32_t>(imm));
}

void x64_emitter::cmp_r64_r64(gpr dst, gpr src)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	const uint8_t rex = static_cast<uint8_t>(0x48 | (((s >> 3) & 1) << 2) | ((d >> 3) & 1));
	buf_.emit_u8(rex);
	buf_.emit_u8(0x39);
	buf_.emit_u8(modrm(0b11, s & 7, d & 7));
}

void x64_emitter::lea_r32_base_index(gpr dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t b = static_cast<uint8_t>(base);
	const uint8_t i = static_cast<uint8_t>(index);
	assert(d < 8 && b < 8 && i < 8);
	assert(i != static_cast<uint8_t>(gpr::rsp));

	buf_.emit_u8(0x8D);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b00, i, b));
}

void x64_emitter::push_r64(gpr r)
{
	const uint8_t v = static_cast<uint8_t>(r);
	assert(v < 8);
	buf_.emit_u8(static_cast<uint8_t>(0x50 + v));
}

void x64_emitter::pop_r64(gpr r)
{
	const uint8_t v = static_cast<uint8_t>(r);
	assert(v < 8);
	buf_.emit_u8(static_cast<uint8_t>(0x58 + v));
}

void x64_emitter::jl_rel8(const int8_t disp)
{
	buf_.emit_u8(0x7C);
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

void x64_emitter::jl_rel32(const int32_t disp)
{
	buf_.emit_u8(0x0F);
	buf_.emit_u8(0x8C);
	buf_.emit_u32(static_cast<uint32_t>(disp));
}

void x64_emitter::ret()
{
	buf_.emit_u8(0xC3);
}

void x64_emitter::emit_win64_arg_shuffle(const int n_int_args)
{
#if defined(_WIN32)
	(void)n_int_args; // Win64 host: kernel ABI already matches.
#else
	// SysV→Win64 int-arg mapping:
	//   rdi -> rcx,  rsi -> rdx,  rdx -> r8,  rcx -> r9
	// Emit in reverse so we don't clobber rdx/rcx before reading them.
	assert(n_int_args >= 0 && n_int_args <= 4);
	if (n_int_args >= 4)
	{
		// mov r9, rcx   : REX.WB(49) 89 /r  ModRM 11 001 001 (C9)
		buf_.emit_u8(0x49);
		buf_.emit_u8(0x89);
		buf_.emit_u8(0xC9);
	}
	if (n_int_args >= 3)
	{
		// mov r8, rdx   : 49 89 D0
		buf_.emit_u8(0x49);
		buf_.emit_u8(0x89);
		buf_.emit_u8(0xD0);
	}
	if (n_int_args >= 2)
	{
		// mov rdx, rsi  : 48 89 F2
		buf_.emit_u8(0x48);
		buf_.emit_u8(0x89);
		buf_.emit_u8(0xF2);
	}
	if (n_int_args >= 1)
	{
		// mov rcx, rdi  : 48 89 F9
		buf_.emit_u8(0x48);
		buf_.emit_u8(0x89);
		buf_.emit_u8(0xF9);
	}
#endif
}
