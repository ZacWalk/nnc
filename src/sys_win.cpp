// nnc — Windows implementation of the sys.h OS abstraction.
// Built only on _WIN32; sys_linux.cpp covers POSIX.

#include "sys.h"

#if defined(_WIN32)

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <intrin.h>

#if defined(_MSC_VER)
#include <stdlib.h> // _set_error_mode / _set_abort_behavior
#endif

void sys_init_console()
{
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
}

void sys_init_crash_handlers()
{
#if defined(_MSC_VER) && defined(_DEBUG)
	_set_error_mode(_OUT_TO_STDERR);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
}

size_t sys_page_size()
{
	static const size_t cached = []
	{
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		return si.dwPageSize
			       ? static_cast<size_t>(si.dwPageSize)
			       : size_t{4096};
	}();
	return cached;
}

void* sys_alloc_exec_pages(const size_t bytes)
{
	void* p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE,
	                       PAGE_READWRITE);
	if (!p)
	{
		std::fprintf(stderr, "sys: VirtualAlloc(%zu) failed (err=%lu)\n",
		             bytes, GetLastError());
	}
	return p;
}

static void protect_or_die(void* p, const size_t bytes, const DWORD prot)
{
	DWORD old = 0;
	if (!VirtualProtect(p, bytes, prot, &old))
	{
		std::fprintf(stderr, "sys: VirtualProtect(%lu) failed (err=%lu)\n",
		             prot, GetLastError());
		std::abort();
	}
}

void sys_protect_rw(void* p, const size_t bytes) { protect_or_die(p, bytes, PAGE_READWRITE); }
void sys_protect_rx(void* p, const size_t bytes) { protect_or_die(p, bytes, PAGE_EXECUTE_READ); }

void sys_flush_icache(const void* p, const size_t bytes)
{
	FlushInstructionCache(GetCurrentProcess(), p, bytes);
}

// ---- File mapping ------------------------------------------------------

bool sys_mmap_file_ro(const char* path, sys_mmap& out)
{
	out = {};

	const int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
	std::vector<wchar_t> wpath(wlen > 0 ? wlen : 1);
	if (wlen > 0) MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

	const HANDLE hFile = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr,
	                                 OPEN_EXISTING,
	                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
	                                 nullptr);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		std::fprintf(stderr, "sys: CreateFileW failed for '%s' (err=%lu)\n",
		             path, GetLastError());
		return false;
	}
	LARGE_INTEGER sz{};
	if (!GetFileSizeEx(hFile, &sz))
	{
		std::fprintf(stderr, "sys: GetFileSizeEx failed (err=%lu)\n", GetLastError());
		CloseHandle(hFile);
		return false;
	}
	const HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!hMap)
	{
		std::fprintf(stderr, "sys: CreateFileMappingW failed (err=%lu)\n", GetLastError());
		CloseHandle(hFile);
		return false;
	}
	const void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
	if (!base)
	{
		std::fprintf(stderr, "sys: MapViewOfFile failed (err=%lu)\n", GetLastError());
		CloseHandle(hMap);
		CloseHandle(hFile);
		return false;
	}

	out.base = base;
	out.size = static_cast<uint64_t>(sz.QuadPart);
	out.handle = hFile;
	out.handle2 = hMap;
	return true;
}

void sys_munmap(sys_mmap& m)
{
	if (m.base)
	{
		UnmapViewOfFile(m.base);
		m.base = nullptr;
	}
	if (m.handle2)
	{
		CloseHandle(m.handle2);
		m.handle2 = nullptr;
	}
	if (m.handle)
	{
		CloseHandle(m.handle);
		m.handle = nullptr;
	}
	m.size = 0;
}

void sys_prefetch_ro(const void* base, const uint64_t bytes)
{
	using PFV = BOOL (WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
	const HMODULE hk = GetModuleHandleW(L"kernel32.dll");
	if (!hk) return;
	const auto pfv = reinterpret_cast<PFV>(GetProcAddress(hk, "PrefetchVirtualMemory"));
	if (!pfv) return;
	WIN32_MEMORY_RANGE_ENTRY r;
	r.VirtualAddress = const_cast<void*>(base);
	r.NumberOfBytes = bytes;
	pfv(GetCurrentProcess(), 1, &r, 0);
}

// ---- CPUID -------------------------------------------------------------

void sys_cpuid(int regs[4], const int leaf) { __cpuid(regs, leaf); }
void sys_cpuidex(int regs[4], const int leaf, const int sub) { __cpuidex(regs, leaf, sub); }
uint64_t sys_xgetbv(const unsigned int xcr) { return _xgetbv(xcr); }

// ---- Filesystem helpers ------------------------------------------------

std::string sys_home_dir()
{
	wchar_t buf[MAX_PATH];
	DWORD n = GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH) return {};
	const int u8 = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr);
	if (u8 <= 0) return {};
	std::string out(static_cast<size_t>(u8 - 1), '\0');
	WideCharToMultiByte(CP_UTF8, 0, buf, -1, out.data(), u8, nullptr, nullptr);
	return out;
}

static bool ends_with_icase(const std::string& s, const char* suffix)
{
	const size_t n = std::strlen(suffix);
	if (s.size() < n) return false;
	for (size_t i = 0; i < n; ++i)
	{
		char a = s[s.size() - n + i];
		char b = suffix[i];
		if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
		if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
		if (a != b) return false;
	}
	return true;
}

void sys_list_files_recursive(const char* dir, const char* ext,
                              std::vector<std::string>& out)
{
	if (!dir || !*dir) return;

	const std::string base = dir;
	const std::string pattern = base + "\\*";

	const int wlen = MultiByteToWideChar(CP_UTF8, 0, pattern.c_str(), -1, nullptr, 0);
	if (wlen <= 0) return;
	std::vector<wchar_t> wpat(static_cast<size_t>(wlen));
	MultiByteToWideChar(CP_UTF8, 0, pattern.c_str(), -1, wpat.data(), wlen);

	WIN32_FIND_DATAW fd{};
	const HANDLE h = FindFirstFileW(wpat.data(), &fd);
	if (h == INVALID_HANDLE_VALUE) return;

	do
	{
		if (fd.cFileName[0] == L'.' &&
		    (fd.cFileName[1] == 0 || (fd.cFileName[1] == L'.' && fd.cFileName[2] == 0)))
			continue;

		const int u8 = WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1,
		                                   nullptr, 0, nullptr, nullptr);
		if (u8 <= 0) continue;
		std::string name(static_cast<size_t>(u8 - 1), '\0');
		WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name.data(), u8, nullptr, nullptr);

		std::string full = base + "\\" + name;
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			sys_list_files_recursive(full.c_str(), ext, out);
		}
		else if (ends_with_icase(name, ext))
		{
			out.push_back(std::move(full));
		}
	}
	while (FindNextFileW(h, &fd));

	FindClose(h);
}

#endif // _WIN32
