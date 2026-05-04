// nnc — executable-memory allocator.
//
// Backed by a process-wide bump-allocator pool of large RWX pages
// (jit_code_pool, internal). Each `commit()` copies the staged bytes
// into the next 16-byte-aligned slot of the active page (allocating a
// fresh page if it doesn't fit) and returns a pointer to executable
// code. `jit_buffer` itself only owns the staging vector; pool pages
// are process-lifetime and never freed.

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

	// Copy the staging bytes into the shared executable-page pool and
	// return a pointer to the start of the placed code. Subsequent
	// commit() calls install into a fresh slot; previously-returned
	// pointers remain valid for the lifetime of the process.
	void* commit();

private:
	std::vector<uint8_t> staging_;
};
