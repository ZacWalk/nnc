// nnc — executable-memory allocator implementation.
//
// Process-wide bump-allocator pool: each `jit_buffer::commit()` copies
// its staging bytes into the next 16-byte-aligned slot of the active
// page. Pages are kept for the lifetime of the process.
//
// W^X discipline: pages live in their steady state as PAGE_EXECUTE_READ
// and are only flipped to PAGE_READWRITE for the brief window during
// which `commit()` is copying new bytes in. Modern Intel CPUs treat
// permanently-RWX pages as potentially self-modifying, which weakens
// the uop cache / iTLB behaviour for hot kernels — keeping pages RX
// during execution avoids that penalty. The two extra VirtualProtect
// calls per commit are paid once per kernel at startup; the kernel
// cache means none happen during a generate run.
//
// Packing many small kernels (dot, gemv, ...) into the same page also
// drops per-kernel overhead from ~4 KB (one page each, 95 % wasted) to
// ~16 bytes (alignment padding only) and keeps related kernels
// physically close, which reduces i-TLB pressure.

#include "jit_buffer.h"

#include "sys.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

jit_buffer::jit_buffer() = default;
jit_buffer::~jit_buffer() = default;

void jit_buffer::append(const void* bytes, const size_t len)
{
	const auto p = static_cast<const uint8_t*>(bytes);
	staging_.insert(staging_.end(), p, p + len);
}

namespace
{
	constexpr size_t POOL_PAGE_BYTES = 64 * 1024; // default page size
	constexpr size_t KERNEL_ALIGN = 16; // 16-byte align each slot

	struct jit_code_pool
	{
		std::mutex mu;
		uint8_t* page = nullptr; // current active page (RX in steady state)
		size_t size = 0; // active page size in bytes
		size_t used = 0; // bytes used in active page

		// Allocate a fresh region of at least `min_bytes`, rounded up to
		// the OS page size. Initial protection is PAGE_READWRITE so the
		// first commit can write without an extra flip; subsequent
		// commits flip RX→RW→RX around the memcpy. The previous active
		// page is leaked into the pool (kept RX executable forever —
		// that is the point of the pool).
		void new_page(const size_t min_bytes)
		{
			const size_t os_page = sys_page_size();
			size_t want = (min_bytes < POOL_PAGE_BYTES) ? POOL_PAGE_BYTES : min_bytes;
			want = (want + os_page - 1) & ~(os_page - 1);
			void* p = sys_alloc_exec_pages(want);
			if (!p)
			{
				std::abort();
			}
			page = static_cast<uint8_t*>(p);
			size = want;
			used = 0;
		}

		// Reserve `len` bytes (aligned), copy `bytes` into them, flip
		// the active page back to RX, flush the icache, return slot ptr.
		void* commit_bytes(const void* bytes, const size_t len)
		{
			std::lock_guard<std::mutex> lk(mu);

			// Align the next slot.
			used = (used + (KERNEL_ALIGN - 1)) & ~(KERNEL_ALIGN - 1);
			const bool need_new = (page == nullptr || used + len > size);
			if (need_new)
			{
				new_page(len); // new page is RW
			}
			else
			{
				// Existing page is RX; flip to RW for the write.
				sys_protect_rw(page, size);
			}

			void* dst = page + used;
			std::memcpy(dst, bytes, len);
			used += len;

			// Restore RX. Doing the whole page (not just the slot)
			// keeps the page in a single VAD region.
			sys_protect_rx(page, size);
			sys_flush_icache(dst, len);
			return dst;
		}
	};

	jit_code_pool& pool()
	{
		static jit_code_pool p;
		return p;
	}
}

void* jit_buffer::commit()
{
	if (staging_.empty()) return nullptr;
	return pool().commit_bytes(staging_.data(), staging_.size());
}
