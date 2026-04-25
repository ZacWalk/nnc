// nnc — minimal GGUF file inspector.
//
// Reads the GGUF v2/v3 header, all metadata key/value pairs, and the
// tensor descriptor table. Does not load tensor data. Intended to back
// the `--gguf-info` CLI mode so we can survey unfamiliar models before
// wiring them into the runtime.
//
// Spec reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum gguf_value_type : uint32_t
{
	GGUF_TYPE_UINT8 = 0,
	GGUF_TYPE_INT8 = 1,
	GGUF_TYPE_UINT16 = 2,
	GGUF_TYPE_INT16 = 3,
	GGUF_TYPE_UINT32 = 4,
	GGUF_TYPE_INT32 = 5,
	GGUF_TYPE_FLOAT32 = 6,
	GGUF_TYPE_BOOL = 7,
	GGUF_TYPE_STRING = 8,
	GGUF_TYPE_ARRAY = 9,
	GGUF_TYPE_UINT64 = 10,
	GGUF_TYPE_INT64 = 11,
	GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_value
{
	gguf_value_type type;

	// scalar storage (whichever matches `type`)
	uint64_t u64; // also used for u8/u16/u32/bool
	int64_t i64; // also used for i8/i16/i32
	double f64; // also used for f32
	std::string str;

	// array storage (when type == GGUF_TYPE_ARRAY)
	gguf_value_type arr_type;
	std::vector<gguf_value> arr;
};

struct gguf_kv
{
	std::string key;
	gguf_value value;
};

struct gguf_tensor_info
{
	std::string name;
	uint32_t n_dims;
	uint64_t ne[4]; // GGUF caps at 4 dims
	uint32_t ggml_type;
	uint64_t offset; // offset within the tensor data block
};

struct gguf_file
{
	uint32_t version;
	uint64_t tensor_count;
	uint64_t kv_count;
	uint64_t alignment; // from general.alignment, default 32
	uint64_t data_offset; // absolute file offset of the tensor data block
	std::vector<gguf_kv> kv;
	std::vector<gguf_tensor_info> tensors;

	// Set by gguf_mmap(); zero/nullptr if only gguf_load() was used.
	void* mapping_handle = nullptr; // HANDLE from CreateFileMappingW
	void* file_handle = nullptr; // HANDLE from CreateFileW
	const uint8_t* mapped_base = nullptr; // start of mmapped region
	uint64_t mapped_size = 0;
};

// Parse the GGUF header + KV table + tensor descriptors. Tensor *data*
// is not read. Returns true on success; on failure prints to stderr.
bool gguf_load(const std::string& path, gguf_file& out);

// Open the file via Win32 CreateFileMapping/MapViewOfFile and parse the
// header into `out`. The mapped region remains valid until
// gguf_unmap(). After this returns, gguf_tensor_data(out, i) yields a
// pointer into the mapped tensor data block.
bool gguf_mmap(const std::string& path, gguf_file& out);
void gguf_unmap(gguf_file& out);

// Pointer to the start of tensor `i`'s raw bytes within the mmapped
// data block. Only valid after gguf_mmap. Returns nullptr otherwise.
const void* gguf_tensor_data(const gguf_file& f, size_t i);

// Find a tensor by exact name. Returns SIZE_MAX if not found.
size_t gguf_find_tensor(const gguf_file& f, const std::string& name);

// Total element count for a tensor (product of all `ne`). Uses uint64
// since some Gemma tables (e.g. per_layer_token_embd) exceed 2^31.
uint64_t gguf_tensor_nelements(const gguf_tensor_info& t);

// Storage size in bytes (post-block-quantization aware for Q* types).
uint64_t gguf_tensor_nbytes(const gguf_tensor_info& t);

// Human-readable name for a GGML tensor type code. Returns "?(<n>)" for
// values we don't recognise.
const char* gguf_ggml_type_name(uint32_t t);

// Find a metadata KV by exact key. Returns nullptr if not present.
const gguf_value* gguf_find_kv(const gguf_file& f, const std::string& key);

// Print a one-line scalar/string value. Arrays are summarised. Long
// strings are truncated.
void gguf_print_value(const gguf_value& v);

// Pretty-print everything we parsed from `f` to stdout.
void gguf_print_info(const gguf_file& f);
