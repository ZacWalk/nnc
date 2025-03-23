// nnc — executable-memory allocator.
// Reserve+commit pages with VirtualAlloc as RW, append bytes, then flip to
// PAGE_EXECUTE_READ via VirtualProtect and FlushInstructionCache.

#pragma once

#include <cstdint>
#include <vector>

class jit_buffer
{
public:
	jit_buffer();
	~jit_buffer();

	jit_buffer(const jit_buffer&) = delete;
	jit_buffer& operator=(const jit_buffer&) = delete;

	// Append raw bytes to the staging vector. Nothing is executable yet.
	void append(const void* bytes, size_t len);
	void emit_u8(const uint8_t b) { append(&b, 1); }
	void emit_u32(const uint32_t v) { append(&v, 4); }
	void emit_u64(const uint64_t v) { append(&v, 8); }

	size_t size() const { return staging_.size(); }
	const uint8_t* staging_data() const { return staging_.data(); }

	// Commit staging to a fresh executable page and return a pointer to the
	// start of the code. Subsequent calls to commit() allocate a new page;
	// the previously-returned pointer is invalidated when this jit_buffer
	// dies (or commit() is called again).
	void* commit();

private:
	void release();

	std::vector<uint8_t> staging_;
	void* exec_ = nullptr;
	size_t exec_size_ = 0;
};
