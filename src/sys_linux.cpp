// nnc — Linux (POSIX) implementation of the sys.h OS abstraction.
// Built only when not _WIN32.

#include "sys.h"

#if !defined(_WIN32)

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include <cpuid.h>

#include <string>
#include <vector>

void sys_init_console()
{
} // Linux terminals are already UTF-8.
void sys_init_crash_handlers()
{
} // glibc doesn't pop modal dialogs on abort().

size_t sys_page_size()
{
	static const size_t cached = []
	{
		const long ps = sysconf(_SC_PAGESIZE);
		return ps > 0 ? static_cast<size_t>(ps) : size_t{4096};
	}();
	return cached;
}

void* sys_alloc_exec_pages(size_t bytes)
{
	void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
	               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED)
	{
		std::fprintf(stderr, "sys: mmap(%zu) failed: %s\n",
		             bytes, std::strerror(errno));
		return nullptr;
	}
	return p;
}

static void protect_or_die(void* p, size_t bytes, int prot)
{
	if (mprotect(p, bytes, prot) != 0)
	{
		std::fprintf(stderr, "sys: mprotect(%d) failed: %s\n",
		             prot, std::strerror(errno));
		std::abort();
	}
}

void sys_protect_rw(void* p, size_t bytes) { protect_or_die(p, bytes, PROT_READ | PROT_WRITE); }
void sys_protect_rx(void* p, size_t bytes) { protect_or_die(p, bytes, PROT_READ | PROT_EXEC); }

void sys_flush_icache(const void* p, size_t bytes)
{
	// On x86 the icache is coherent with stores from the same core, so
	// this is effectively a no-op. We still call the GCC builtin to keep
	// the contract clean (and to be correct on other archs in future).
	auto* s = const_cast<char*>(static_cast<const char*>(p));
	__builtin___clear_cache(s, s + bytes);
}

// ---- File mapping ------------------------------------------------------

bool sys_mmap_file_ro(const char* path, sys_mmap& out)
{
	out = {};

	const int fd = open(path, O_RDONLY);
	if (fd < 0)
	{
		std::fprintf(stderr, "sys: open('%s') failed: %s\n",
		             path, std::strerror(errno));
		return false;
	}
	struct stat st{};
	if (fstat(fd, &st) != 0)
	{
		std::fprintf(stderr, "sys: fstat failed: %s\n", std::strerror(errno));
		close(fd);
		return false;
	}
	const size_t len = static_cast<size_t>(st.st_size);
	void* base = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
	if (base == MAP_FAILED)
	{
		std::fprintf(stderr, "sys: mmap('%s', %zu) failed: %s\n",
		             path, len, std::strerror(errno));
		close(fd);
		return false;
	}

	out.base = base;
	out.size = static_cast<uint64_t>(st.st_size);
	out.handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
	out.handle2 = nullptr;
	return true;
}

void sys_munmap(sys_mmap& m)
{
	if (m.base && m.size)
	{
		munmap(const_cast<void*>(m.base), static_cast<size_t>(m.size));
	}
	if (m.handle)
	{
		const int fd = static_cast<int>(reinterpret_cast<intptr_t>(m.handle));
		close(fd);
	}
	m.base = nullptr;
	m.size = 0;
	m.handle = nullptr;
	m.handle2 = nullptr;
}

void sys_prefetch_ro(const void* base, uint64_t bytes)
{

#if defined(POSIX_MADV_WILLNEED)
posix_madvise (const_cast<void*>(base), static_cast<size_t>(bytes),
                                                           POSIX_MADV_WILLNEED);
#else
madvise (const_cast<void*>(base), static_cast<size_t>(bytes), MADV_WILLNEED);
#endif
}

// ---- CPUID -------------------------------------------------------------

void sys_cpuid(int regs[4], int leaf)
{
	unsigned int a = 0, b = 0, c = 0, d = 0;
	__get_cpuid(static_cast<unsigned int>(leaf), &a, &b, &c, &d);
	regs[0] = static_cast<int>(a);
	regs[1] = static_cast<int>(b);
	regs[2] = static_cast<int>(c);
	regs[3] = static_cast<int>(d);
}

void sys_cpuidex(int regs[4], int leaf, int sub)
{
	unsigned int a = 0, b = 0, c = 0, d = 0;
	__cpuid_count(static_cast<unsigned int>(leaf),
	              static_cast<unsigned int>(sub), a, b, c, d);
	regs[0] = static_cast<int>(a);
	regs[1] = static_cast<int>(b);
	regs[2] = static_cast<int>(c);
	regs[3] = static_cast<int>(d);
}

uint64_t sys_xgetbv(unsigned int xcr)
{
	unsigned int eax = 0, edx = 0;
	__asm__ volatile (
	"xgetbv"
	:
	"=a"(eax), "=d"(edx)
	:
	"c"(xcr)
	)
	;
	return (static_cast<uint64_t>(edx) << 32) | eax;
}

// ---- Filesystem helpers ------------------------------------------------

std::string sys_home_dir()
{
	const char* h = std::getenv("HOME");
	return h ? std::string(h) : std::string();
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
	DIR* d = opendir(dir);
	if (!d) return;

	const std::string base = dir;
	while (struct dirent* de = readdir(d))
	{
		const char* name = de->d_name;
		if (name[0] == '.' && (name[1] == 0 || (name[1] == '.' && name[2] == 0)))
			continue;

		std::string full = base + "/" + name;

		struct stat st{};
		if (stat(full.c_str(), &st) != 0) continue;

		if (S_ISDIR(st.st_mode))
		{
			sys_list_files_recursive(full.c_str(), ext, out);
		}
		else if (S_ISREG(st.st_mode) && ends_with_icase(full, ext))
		{
			out.push_back(std::move(full));
		}
	}
	closedir(d);
}

#endif // !_WIN32
