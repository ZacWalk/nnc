// nnc — minimal GGUF file inspector. See gguf.h for the format reference.

#include "gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace
{
	constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

	// Bound the per-allocation sizes we accept from the file. Anything
	// larger almost certainly means a corrupt or hostile input. The
	// limits are generous enough for real models (256k-vocab string
	// arrays etc.).
	constexpr uint64_t MAX_STRING_LEN = 1ull << 24; // 16 MiB
	constexpr uint64_t MAX_ARRAY_LEN = 1ull << 28; // 256 M elements
	constexpr uint64_t MAX_TENSOR_COUNT = 1ull << 20; // 1 M tensors
	constexpr uint64_t MAX_KV_COUNT = 1ull << 20;

	struct reader
	{
		std::ifstream& s;
		const std::string& path;
		bool ok = true;

		bool read_raw(void* dst, const size_t n)
		{
			if (!ok)
				return false;
			s.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
			if (!s)
			{
				fprintf(stderr, "gguf: short read of %zu bytes from '%s'\n", n, path.c_str());
				ok = false;
				return false;
			}
			return true;
		}

		template <typename T>
		bool read_pod(T& v)
		{
			return read_raw(&v, sizeof(T));
		}

		bool read_string(std::string& out)
		{
			uint64_t len = 0;
			if (!read_pod(len))
				return false;
			if (len > MAX_STRING_LEN)
			{
				fprintf(stderr, "gguf: refusing string of length %" PRIu64 " in '%s'\n",
				        len, path.c_str());
				ok = false;
				return false;
			}
			out.resize(len);
			if (len > 0 && !read_raw(out.data(), len))
				return false;
			return true;
		}
	};

	bool read_value(reader& r, gguf_value_type type, gguf_value& out);

	bool read_scalar(reader& r, const gguf_value_type type, gguf_value& out)
	{
		out.type = type;
		switch (type)
		{
		case GGUF_TYPE_UINT8:
			{
				uint8_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.u64 = v;
				return true;
			}
		case GGUF_TYPE_INT8:
			{
				int8_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.i64 = v;
				return true;
			}
		case GGUF_TYPE_UINT16:
			{
				uint16_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.u64 = v;
				return true;
			}
		case GGUF_TYPE_INT16:
			{
				int16_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.i64 = v;
				return true;
			}
		case GGUF_TYPE_UINT32:
			{
				uint32_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.u64 = v;
				return true;
			}
		case GGUF_TYPE_INT32:
			{
				int32_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.i64 = v;
				return true;
			}
		case GGUF_TYPE_UINT64:
			{
				uint64_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.u64 = v;
				return true;
			}
		case GGUF_TYPE_INT64:
			{
				int64_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.i64 = v;
				return true;
			}
		case GGUF_TYPE_FLOAT32:
			{
				float v = 0;
				if (!r.read_pod(v))
					return false;
				out.f64 = v;
				return true;
			}
		case GGUF_TYPE_FLOAT64:
			{
				double v = 0;
				if (!r.read_pod(v))
					return false;
				out.f64 = v;
				return true;
			}
		case GGUF_TYPE_BOOL:
			{
				uint8_t v = 0;
				if (!r.read_pod(v))
					return false;
				out.u64 = (v != 0) ? 1 : 0;
				return true;
			}
		case GGUF_TYPE_STRING:
			return r.read_string(out.str);
		default:
			fprintf(stderr, "gguf: unexpected scalar value type %u\n",
			        static_cast<unsigned>(type));
			r.ok = false;
			return false;
		}
	}

	bool read_value(reader& r, const gguf_value_type type, gguf_value& out)
	{
		if (type == GGUF_TYPE_ARRAY)
		{
			out.type = GGUF_TYPE_ARRAY;

			uint32_t arr_type_u32 = 0;
			uint64_t len = 0;
			if (!r.read_pod(arr_type_u32) || !r.read_pod(len))
				return false;
			if (len > MAX_ARRAY_LEN)
			{
				fprintf(stderr, "gguf: refusing array of length %" PRIu64 "\n", len);
				r.ok = false;
				return false;
			}

			out.arr_type = static_cast<gguf_value_type>(arr_type_u32);
			out.arr.resize(len);
			for (uint64_t i = 0; i < len; ++i)
			{
				if (!read_value(r, out.arr_type, out.arr[i]))
					return false;
			}
			return true;
		}

		return read_scalar(r, type, out);
	}
}

const char* gguf_ggml_type_name(const uint32_t t)
{
	// Mirrors enum ggml_type as of llama.cpp / ggml master. Covers the
	// subset we are likely to see in modern model dumps.
	switch (t)
	{
	case 0: return "F32";
	case 1: return "F16";
	case 2: return "Q4_0";
	case 3: return "Q4_1";
	case 6: return "Q5_0";
	case 7: return "Q5_1";
	case 8: return "Q8_0";
	case 9: return "Q8_1";
	case 10: return "Q2_K";
	case 11: return "Q3_K";
	case 12: return "Q4_K";
	case 13: return "Q5_K";
	case 14: return "Q6_K";
	case 15: return "Q8_K";
	case 16: return "IQ2_XXS";
	case 17: return "IQ2_XS";
	case 18: return "IQ3_XXS";
	case 19: return "IQ1_S";
	case 20: return "IQ4_NL";
	case 21: return "IQ3_S";
	case 22: return "IQ2_S";
	case 23: return "IQ4_XS";
	case 24: return "I8";
	case 25: return "I16";
	case 26: return "I32";
	case 27: return "I64";
	case 28: return "F64";
	case 29: return "IQ1_M";
	case 30: return "BF16";
	case 31: return "Q4_0_4_4";
	case 32: return "Q4_0_4_8";
	case 33: return "Q4_0_8_8";
	case 34: return "TQ1_0";
	case 35: return "TQ2_0";
	}
	thread_local char buf[24];
	snprintf(buf, sizeof(buf), "?(%u)", t);
	return buf;
}

bool gguf_load(const std::string& path, gguf_file& out)
{
	std::ifstream s(path, std::ios::binary);
	if (!s)
	{
		fprintf(stderr, "gguf: cannot open '%s'\n", path.c_str());
		return false;
	}

	reader r{s, path};

	uint32_t magic = 0;
	if (!r.read_pod(magic))
		return false;
	if (magic != GGUF_MAGIC)
	{
		fprintf(stderr, "gguf: '%s' is not a GGUF file (magic=0x%08x)\n",
		        path.c_str(), magic);
		return false;
	}

	if (!r.read_pod(out.version))
		return false;
	if (out.version != 2 && out.version != 3)
	{
		fprintf(stderr, "gguf: unsupported version %u in '%s' (expected 2 or 3)\n",
		        out.version, path.c_str());
		return false;
	}

	if (!r.read_pod(out.tensor_count) || !r.read_pod(out.kv_count))
		return false;
	if (out.tensor_count > MAX_TENSOR_COUNT || out.kv_count > MAX_KV_COUNT)
	{
		fprintf(stderr, "gguf: implausible counts (tensors=%" PRIu64 ", kv=%" PRIu64 ")\n",
		        out.tensor_count, out.kv_count);
		return false;
	}

	out.kv.resize(out.kv_count);
	for (size_t i = 0; i < out.kv.size(); ++i)
	{
		auto& kv = out.kv[i];
		uint32_t type_u32 = 0;
		if (!r.read_string(kv.key) || !r.read_pod(type_u32))
			return false;
		if (!read_value(r, static_cast<gguf_value_type>(type_u32), kv.value))
			return false;
	}

	out.alignment = 32;
	for (const auto& kv : out.kv)
	{
		if (kv.key == "general.alignment" && kv.value.type == GGUF_TYPE_UINT32)
		{
			out.alignment = kv.value.u64 ? kv.value.u64 : 32;
			break;
		}
	}

	out.tensors.resize(out.tensor_count);
	for (size_t i = 0; i < out.tensors.size(); ++i)
	{
		auto& t = out.tensors[i];
		if (!r.read_string(t.name) || !r.read_pod(t.n_dims))
			return false;
		if (t.n_dims == 0 || t.n_dims > 4)
		{
			fprintf(stderr, "gguf: tensor '%s' has bad n_dims=%u\n",
			        t.name.c_str(), t.n_dims);
			return false;
		}
		t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 1;
		for (uint32_t d = 0; d < t.n_dims; ++d)
		{
			if (!r.read_pod(t.ne[d]))
				return false;
		}
		if (!r.read_pod(t.ggml_type) || !r.read_pod(t.offset))
			return false;
	}

	// Align to `alignment` past the descriptor table; that absolute
	// offset is where the tensor data block starts.
	const uint64_t pos = s.tellg();
	const uint64_t align = out.alignment;
	out.data_offset = (pos + align - 1) / align * align;

	return true;
}

namespace
{
	// Block layout for the ggml types we know about. {block_size_bytes, elements_per_block}.
	// 0 size means "unknown / unsupported by this build" (we'll bail with -1).
	struct ggml_block_info
	{
		uint32_t bytes;
		uint32_t elems;
	};

	ggml_block_info ggml_block_for(const uint32_t t)
	{
		switch (t)
		{
		case 0: return {4, 1}; // F32
		case 1: return {2, 1}; // F16
		case 2: return {18, 32}; // Q4_0   (2 + 16)
		case 3: return {20, 32}; // Q4_1   (2+2+16)
		case 6: return {22, 32}; // Q5_0
		case 7: return {24, 32}; // Q5_1
		case 8: return {34, 32}; // Q8_0
		case 9: return {40, 32}; // Q8_1
		case 24: return {1, 1}; // I8
		case 25: return {2, 1}; // I16
		case 26: return {4, 1}; // I32
		case 27: return {8, 1}; // I64
		case 28: return {8, 1}; // F64
		case 30: return {2, 1}; // BF16
		default: return {0, 0};
		}
	}
}

uint64_t gguf_tensor_nelements(const gguf_tensor_info& t)
{
	uint64_t n = 1;
	for (uint32_t d = 0; d < t.n_dims; ++d) n *= t.ne[d];
	return n;
}

uint64_t gguf_tensor_nbytes(const gguf_tensor_info& t)
{
	const auto bi = ggml_block_for(t.ggml_type);
	if (bi.bytes == 0) return 0;
	const uint64_t n = gguf_tensor_nelements(t);
	return (n / bi.elems) * bi.bytes;
}

size_t gguf_find_tensor(const gguf_file& f, const std::string& name)
{
	for (size_t i = 0; i < f.tensors.size(); ++i)
		if (f.tensors[i].name == name) return i;
	return SIZE_MAX;
}

const gguf_value* gguf_find_kv(const gguf_file& f, const std::string& key)
{
	for (const auto& kv : f.kv)
		if (kv.key == key) return &kv.value;
	return nullptr;
}

bool gguf_mmap(const std::string& path, gguf_file& out)
{
	if (!gguf_load(path, out)) return false;

	// Convert UTF-8 path to wide for CreateFileW.
	const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
	std::vector<wchar_t> wpath(wlen > 0 ? wlen : 1);
	if (wlen > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

	const HANDLE hFile = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr,
	                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
	                                 nullptr);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		fprintf(stderr, "gguf: CreateFileW failed for '%s' (err=%lu)\n",
		        path.c_str(), GetLastError());
		return false;
	}
	LARGE_INTEGER sz{};
	if (!GetFileSizeEx(hFile, &sz))
	{
		CloseHandle(hFile);
		fprintf(stderr, "gguf: GetFileSizeEx failed (err=%lu)\n", GetLastError());
		return false;
	}
	const HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!hMap)
	{
		CloseHandle(hFile);
		fprintf(stderr, "gguf: CreateFileMappingW failed (err=%lu)\n", GetLastError());
		return false;
	}
	const void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
	if (!base)
	{
		CloseHandle(hMap);
		CloseHandle(hFile);
		fprintf(stderr, "gguf: MapViewOfFile failed (err=%lu)\n", GetLastError());
		return false;
	}
	out.file_handle = hFile;
	out.mapping_handle = hMap;
	out.mapped_base = static_cast<const uint8_t*>(base);
	out.mapped_size = static_cast<uint64_t>(sz.QuadPart);
	return true;
}

void gguf_unmap(gguf_file& out)
{
	if (out.mapped_base)
	{
		UnmapViewOfFile(out.mapped_base);
		out.mapped_base = nullptr;
	}
	if (out.mapping_handle)
	{
		CloseHandle(out.mapping_handle);
		out.mapping_handle = nullptr;
	}
	if (out.file_handle)
	{
		CloseHandle(out.file_handle);
		out.file_handle = nullptr;
	}
	out.mapped_size = 0;
}

const void* gguf_tensor_data(const gguf_file& f, const size_t i)
{
	if (!f.mapped_base || i >= f.tensors.size()) return nullptr;
	const uint64_t off = f.data_offset + f.tensors[i].offset;
	if (off >= f.mapped_size) return nullptr;
	return f.mapped_base + off;
}

void gguf_print_value(const gguf_value& v)
{
	switch (v.type)
	{
	case GGUF_TYPE_UINT8:
	case GGUF_TYPE_UINT16:
	case GGUF_TYPE_UINT32:
	case GGUF_TYPE_UINT64:
		printf("%" PRIu64, v.u64);
		break;
	case GGUF_TYPE_INT8:
	case GGUF_TYPE_INT16:
	case GGUF_TYPE_INT32:
	case GGUF_TYPE_INT64:
		printf("%" PRId64, v.i64);
		break;
	case GGUF_TYPE_FLOAT32:
	case GGUF_TYPE_FLOAT64:
		printf("%g", v.f64);
		break;
	case GGUF_TYPE_BOOL:
		printf("%s", v.u64 ? "true" : "false");
		break;
	case GGUF_TYPE_STRING:
		{
			// Truncate long strings (e.g. chat templates) for readability.
			constexpr size_t MAX = 96;
			if (v.str.size() <= MAX)
				printf("\"%s\"", v.str.c_str());
			else
				printf("\"%.*s...\" (%zu chars)",
				       static_cast<int>(MAX), v.str.c_str(), v.str.size());
			break;
		}
	case GGUF_TYPE_ARRAY:
		{
			auto etn = "?";
			switch (v.arr_type)
			{
			case GGUF_TYPE_UINT8: etn = "u8";
				break;
			case GGUF_TYPE_INT8: etn = "i8";
				break;
			case GGUF_TYPE_UINT16: etn = "u16";
				break;
			case GGUF_TYPE_INT16: etn = "i16";
				break;
			case GGUF_TYPE_UINT32: etn = "u32";
				break;
			case GGUF_TYPE_INT32: etn = "i32";
				break;
			case GGUF_TYPE_UINT64: etn = "u64";
				break;
			case GGUF_TYPE_INT64: etn = "i64";
				break;
			case GGUF_TYPE_FLOAT32: etn = "f32";
				break;
			case GGUF_TYPE_FLOAT64: etn = "f64";
				break;
			case GGUF_TYPE_BOOL: etn = "bool";
				break;
			case GGUF_TYPE_STRING: etn = "string";
				break;
			case GGUF_TYPE_ARRAY: etn = "array";
				break;
			}
			printf("array<%s>[%zu]", etn, v.arr.size());
			// (Show first few scalar elements; skip nested arrays.)
			const size_t show = v.arr.size() < 4 ? v.arr.size() : 4;
			if (show > 0 && v.arr_type != GGUF_TYPE_ARRAY)
			{
				printf(" = { ");
				for (size_t i = 0; i < show; ++i)
				{
					if (i) printf(", ");
					gguf_print_value(v.arr[i]);
				}
				if (v.arr.size() > show)
					printf(", ...");
				printf(" }");
			}
			break;
		}
	}
}

void gguf_print_info(const gguf_file& f)
{
	printf("gguf: version=%u, kv=%" PRIu64 ", tensors=%" PRIu64
	       ", alignment=%" PRIu64 ", data_offset=0x%" PRIx64 "\n",
	       f.version, f.kv_count, f.tensor_count, f.alignment, f.data_offset);

	printf("\n--- metadata (%zu entries) ---\n", f.kv.size());
	for (const auto& kv : f.kv)
	{
		printf("  %s = ", kv.key.c_str());
		gguf_print_value(kv.value);
		printf("\n");
	}

	printf("\n--- tensors (%zu) ---\n", f.tensors.size());
	// header
	printf("  %-48s %-8s %-28s %16s\n", "name", "type", "shape", "offset");
	uint64_t total_elems = 0;
	for (const auto& t : f.tensors)
	{
		char shape[64];
		if (t.n_dims == 1)
			snprintf(shape, sizeof(shape), "[%" PRIu64 "]", t.ne[0]);
		else if (t.n_dims == 2)
			snprintf(shape, sizeof(shape), "[%" PRIu64 ", %" PRIu64 "]",
			         t.ne[0], t.ne[1]);
		else if (t.n_dims == 3)
			snprintf(shape, sizeof(shape), "[%" PRIu64 ", %" PRIu64 ", %" PRIu64 "]",
			         t.ne[0], t.ne[1], t.ne[2]);
		else
			snprintf(shape, sizeof(shape),
			         "[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "]",
			         t.ne[0], t.ne[1], t.ne[2], t.ne[3]);

		uint64_t n = 1;
		for (uint32_t d = 0; d < t.n_dims; ++d)
			n *= t.ne[d];
		total_elems += n;

		printf("  %-48s %-8s %-28s 0x%14" PRIx64 "\n",
		       t.name.c_str(),
		       gguf_ggml_type_name(t.ggml_type),
		       shape,
		       t.offset);
	}
	printf("\n  total elements across all tensors: %" PRIu64 "\n", total_elems);
}

// ----------------------------------------------------------------------------
// --- gguf_stats CLI helper ---------------------------------------------------
// ----------------------------------------------------------------------------

#include "runtime.h" // nnc_bf16_to_f32_row, nnc_fp16_t conversion (not yet)

namespace
{
	// Convert one element of `type` at byte pointer `p` to float. Returns
	// NaN for unsupported types so the caller can detect them.
	float elem_to_f32(const uint32_t type, const uint8_t* p)
	{
		switch (type)
		{
		case 0: // F32
			{
				float v;
				memcpy(&v, p, 4);
				return v;
			}
		case 1: // F16
			{
				uint16_t h;
				memcpy(&h, p, 2);
				// Inline f16->f32 (avoid pulling in F16C just for this CLI).
				const uint32_t s = (h & 0x8000u) << 16;
				const uint32_t e = (h >> 10) & 0x1fu;
				const uint32_t m = h & 0x3ffu;
				uint32_t u;
				if (e == 0)
					u = s | (m == 0 ? 0u : ((m << 13) | 0x33800000u)); // subnormal -> normal-ish
				else if (e == 31)
					u = s | 0x7f800000u | (m << 13);
				else
					u = s | ((e + 112) << 23) | (m << 13);
				float v;
				memcpy(&v, &u, 4);
				return v;
			}
		case 30: // BF16
			{
				uint16_t b;
				memcpy(&b, p, 2);
				const uint32_t u = static_cast<uint32_t>(b) << 16;
				float v;
				memcpy(&v, &u, 4);
				return v;
			}
		default:
			{
				constexpr auto qn = std::numeric_limits<float>::quiet_NaN();
				(void)qn;
				return std::nanf("");
			}
		}
	}

	bool stats_for_tensor(const gguf_file& f, const size_t idx)
	{
		const auto& t = f.tensors[idx];
		const auto bi = ggml_block_for(t.ggml_type);
		const uint64_t n = gguf_tensor_nelements(t);

		printf("  %-48s %-6s n=%-12" PRIu64,
		       t.name.c_str(), gguf_ggml_type_name(t.ggml_type), n);

		// Only F32 / F16 / BF16 are decoded element-wise here; quantised
		// types would need their dequantiser. Print a placeholder.
		if (t.ggml_type != 0 && t.ggml_type != 1 && t.ggml_type != 30)
		{
			printf("  (no scalar decode for this type)\n");
			return true;
		}

		const auto base = static_cast<const uint8_t*>(gguf_tensor_data(f, idx));
		if (!base)
		{
			printf("  (no data)\n");
			return false;
		}
		const uint32_t bytes_per = bi.bytes; // 1 elem per block here

		// Stream through the tensor accumulating stats.
		double sum = 0.0;
		double sum_sq = 0.0;
		float lo = std::numeric_limits<float>::infinity();
		float hi = -std::numeric_limits<float>::infinity();
		uint64_t nan_count = 0;
		for (uint64_t i = 0; i < n; ++i)
		{
			const float v = elem_to_f32(t.ggml_type, base + i * bytes_per);
			if (std::isnan(v))
			{
				++nan_count;
				continue;
			}
			if (v < lo) lo = v;
			if (v > hi) hi = v;
			sum += v;
			sum_sq += static_cast<double>(v) * v;
		}
		const double mean = n ? sum / static_cast<double>(n) : 0.0;
		const double var = n ? (sum_sq / static_cast<double>(n)) - mean * mean : 0.0;
		const double rms = std::sqrt(var > 0 ? var : 0.0);
		printf("  min=%+.4g max=%+.4g mean=%+.4g rms=%.4g",
		       static_cast<double>(lo), static_cast<double>(hi), mean, rms);
		if (nan_count) printf(" nan=%" PRIu64, nan_count);

		// First few values, for human spot-checking.
		const uint64_t show = n < 6 ? n : 6;
		printf("  first=[");
		for (uint64_t i = 0; i < show; ++i)
		{
			if (i) printf(", ");
			printf("%+.4g", static_cast<double>(elem_to_f32(t.ggml_type, base + i * bytes_per)));
		}
		printf("%s]\n", show < n ? ", ..." : "");
		return true;
	}
}

int gguf_stats_main(const char* path, const char* needle)
{
	gguf_file f{};
	if (!gguf_mmap(path, f)) return 1;

	printf("gguf: '%s' mapped %" PRIu64 " bytes, %" PRIu64 " tensors\n",
	       path, f.mapped_size, f.tensor_count);
	if (needle && *needle)
		printf("gguf: filter = substring '%s'\n", needle);
	printf("\n");

	int matched = 0;
	for (size_t i = 0; i < f.tensors.size(); ++i)
	{
		if (needle && *needle && f.tensors[i].name.find(needle) == std::string::npos)
			continue;
		stats_for_tensor(f, i);
		++matched;
	}
	if (matched == 0)
		printf("(no tensors matched)\n");

	gguf_unmap(f);
	return 0;
}
