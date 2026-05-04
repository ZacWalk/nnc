// nnc — thin OS abstraction layer.
// All Windows / Linux specific calls live behind this header. Implementations
// are in sys_win.cpp (Windows) or sys_linux.cpp (Linux). One of those two
// translation units is built per platform; the other is excluded.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// One-time console / debug setup. Forces UTF-8 output on Windows (so BPE
// pieces with U+2581 etc. render). No-op on Linux (terminals are already
// UTF-8 in practice). Also installs MSVC's "no modal dialogs on assert"
// behaviour in debug builds; no-op elsewhere.
void sys_init_console();
void sys_init_crash_handlers();

// ---- Executable memory pool primitives --------------------------------

// OS page size (e.g. 4096 on x86-64 Windows / Linux).
size_t sys_page_size();

// Reserve and commit `bytes` of writable memory suitable for later being
// flipped to executable. Returns nullptr on failure (caller may abort).
void* sys_alloc_exec_pages(size_t bytes);

// Flip an existing region between writable and executable. `bytes` should
// be a whole number of pages starting at `p`.
void sys_protect_rw(void* p, size_t bytes);
void sys_protect_rx(void* p, size_t bytes);

// Notify the CPU that the bytes at `[p, p+bytes)` were just written and
// will be executed shortly. Required after publishing JITted code.
void sys_flush_icache(const void* p, size_t bytes);

// ---- Read-only file mapping -------------------------------------------

struct sys_mmap
{
	const void* base = nullptr; // start of mapped region (page-aligned)
	uint64_t size = 0; // bytes
	// Opaque OS handles. Windows uses both (file + mapping HANDLE);
	// Linux only needs the file descriptor.
	void* handle = nullptr;
	void* handle2 = nullptr;
};

// mmap `path` read-only into the process. Returns false (and prints a
// reason to stderr) on failure.
bool sys_mmap_file_ro(const char* path, sys_mmap& out);
void sys_munmap(sys_mmap& m);

// Hint the OS to start paging the entire mapping into RAM. Best-effort;
// failure is silently ignored.
void sys_prefetch_ro(const void* base, uint64_t bytes);

// ---- CPUID / XGETBV ----------------------------------------------------

// Cross-compiler CPUID wrappers. `regs` is filled as {eax, ebx, ecx, edx}.
void sys_cpuid(int regs[4], int leaf);
void sys_cpuidex(int regs[4], int leaf, int subleaf);

// Read an extended control register (used to confirm OS YMM state save).
uint64_t sys_xgetbv(unsigned int xcr);

// ---- Filesystem helpers (model picker) --------------------------------

// Returns the user's home directory (Windows: %USERPROFILE%; POSIX: $HOME).
// Empty string if it can't be determined. No trailing separator.
std::string sys_home_dir();

// Recursively scan `dir` for files whose name ends with `ext`
// (case-insensitive, e.g. ".gguf"). Appends full paths to `out`. Silently
// returns on missing directory or any I/O error.
void sys_list_files_recursive(const char* dir, const char* ext,
                              std::vector<std::string>& out);
