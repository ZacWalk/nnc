// nnc — x86-64 instruction encoders.
// Hand-rolled, no tables. Only the instructions nnc actually needs.
// Win64 ABI: int args in RCX, RDX, R8, R9; float return in XMM0;
// integer return in RAX. Volatile: RAX, RCX, RDX, R8-R11, XMM0-XMM5.
// Nonvolatile (must be saved if used): RBX, RBP, RSI, RDI, R12-R15, XMM6-XMM15.

#pragma once

#include <cstdint>

class jit_buffer;

enum class gpr : uint8_t
{
	rax = 0, rcx = 1, rdx = 2, rbx = 3,
	rsp = 4, rbp = 5, rsi = 6, rdi = 7,
	r8 = 8, r9 = 9, r10 = 10, r11 = 11,
	r12 = 12, r13 = 13, r14 = 14, r15 = 15,
};

class x64_emitter
{
public:
	explicit x64_emitter(jit_buffer& buf) : buf_(buf)
	{
	}

	// ---- 32-bit move / xor (low 8 GPRs only) ---------------------------

	// mov r32, imm32   (B8+rd id)
	void mov_r32_imm32(gpr dst, uint32_t imm);

	// xor r32, r32     (31 /r)   zeroes full r64 implicitly.
	void xor_r32_r32(gpr dst, gpr src);

	// ---- 64-bit moves --------------------------------------------------

	// mov r64, r64   (REX.W 89 /r)   src may be extended (r8..r15);
	// dst must be one of the low 8.
	void mov_r64_r64_srcext_ok(gpr dst_low8, gpr src_any);

	// ---- 64-bit arithmetic (low 8 GPR dst) -----------------------------

	// add r64, imm8    (REX.W 83 /0 ib)   sign-extended 8-bit immediate.
	void add_r64_imm8(gpr dst, int8_t imm);

	// add r64, imm32   (REX.W 81 /0 id)
	void add_r64_imm32(gpr dst, int32_t imm);

	// cmp r64, imm32   (REX.W 81 /7 id)
	void cmp_r64_imm32(gpr dst, int32_t imm);

	// cmp r64, r64     (REX.W 39 /r)
	void cmp_r64_r64(gpr dst, gpr src);

	// ---- Address forms -------------------------------------------------

	// lea r32, [base + index*1]   (8D /r, ModRM+SIB)
	void lea_r32_base_index(gpr dst, gpr base, gpr index);

	// ---- Stack ---------------------------------------------------------

	void push_r64(gpr r); // 50+rd  (low 8 only)
	void pop_r64(gpr r); // 58+rd  (low 8 only)

	// ---- ABI shim ------------------------------------------------------

	// On Linux (System V x86-64), shuffle the first n integer arguments
	// from the SysV registers (RDI, RSI, RDX, RCX) into the Windows x64
	// registers (RCX, RDX, R8, R9). On Windows this is a no-op so the
	// rest of every kernel can be written as if Win64 ABI is in force
	// regardless of host OS. n_int_args must be in [0, 4]. Float args
	// land in XMM0..XMM3 in both ABIs and need no shuffle.
	void emit_win64_arg_shuffle(int n_int_args);

	// ---- Control flow --------------------------------------------------

	void jl_rel8(int8_t disp); // 7C cb
	void jl_rel32(int32_t disp); // 0F 8C cd
	void ret(); // C3

private:
	jit_buffer& buf_;
};
