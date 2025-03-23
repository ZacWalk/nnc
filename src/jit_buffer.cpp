// nnc — executable-memory allocator implementation.
// Stages bytes in a std::vector, then on commit() reserves pages with
// VirtualAlloc(RW), copies the bytes in, flips them to PAGE_EXECUTE_READ
// via VirtualProtect, and calls FlushInstructionCache so the CPU sees the
// new code.

#include "jit_buffer.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstdlib>

jit_buffer::jit_buffer() = default;

jit_buffer::~jit_buffer()
{
	release();
}

void jit_buffer::append(const void* bytes, const size_t len)
{
	const auto p = static_cast<const uint8_t*>(bytes);
	staging_.insert(staging_.end(), p, p + len);
}

void jit_buffer::release()
{
	if (exec_)
	{
		VirtualFree(exec_, 0, MEM_RELEASE);
		exec_ = nullptr;
		exec_size_ = 0;
	}
}

void* jit_buffer::commit()
{
	release();

	if (staging_.empty())
	{
		return nullptr;
	}

	SYSTEM_INFO si;
	GetSystemInfo(&si);
	const size_t page = si.dwPageSize ? si.dwPageSize : 4096;
	exec_size_ = (staging_.size() + page - 1) & ~(page - 1);

	exec_ = VirtualAlloc(nullptr, exec_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if (!exec_)
	{
		fprintf(stderr, "jit_buffer: VirtualAlloc failed (err=%lu)\n", GetLastError());
		std::abort();
	}

	std::memcpy(exec_, staging_.data(), staging_.size());

	DWORD old_protect = 0;
	if (!VirtualProtect(exec_, exec_size_, PAGE_EXECUTE_READ, &old_protect))
	{
		fprintf(stderr, "jit_buffer: VirtualProtect failed (err=%lu)\n", GetLastError());
		std::abort();
	}

	FlushInstructionCache(GetCurrentProcess(), exec_, exec_size_);
	return exec_;
}
