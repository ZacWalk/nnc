// nnc — Gemma 3n model loader. See gemma.h for the data structures.

#include "gemma.h"
#include "nn_ops.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace
{
	// GGUF ggml_type code -> nnc_type. Returns NNC_TYPE_COUNT for codes
	// we don't yet handle (the loader treats that as a fatal error).
	nnc_type to_nnc_type(const uint32_t ggml_t)
	{
		switch (ggml_t)
		{
		case 0: return NNC_TYPE_F32;
		case 1: return NNC_TYPE_F16;
		case 26: return NNC_TYPE_I32;
		case 30: return NNC_TYPE_BF16;
		default: return NNC_TYPE_COUNT;
		}
	}

	// Look up an integer hparam from the GGUF metadata. Accepts u32/i32/u64/i64.
	bool kv_int(const gguf_file& f, const char* key, int& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return false;
		switch (v->type)
		{
		case GGUF_TYPE_UINT8:
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_UINT64:
			out = static_cast<int>(v->u64);
			return true;
		case GGUF_TYPE_INT8:
		case GGUF_TYPE_INT16:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_INT64:
			out = static_cast<int>(v->i64);
			return true;
		default: return false;
		}
	}

	bool kv_float(const gguf_file& f, const char* key, float& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return false;
		if (v->type == GGUF_TYPE_FLOAT32 || v->type == GGUF_TYPE_FLOAT64)
		{
			out = static_cast<float>(v->f64);
			return true;
		}
		return false;
	}

	bool kv_str(const gguf_file& f, const char* key, std::string& out)
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v || v->type != GGUF_TYPE_STRING) return false;
		out = v->str;
		return true;
	}

	// Construct an nnc_tensor descriptor from a GGUF tensor index. Data
	// points into the mmap region (no copy). Strides assume row-major
	// contiguous storage. Returns nullptr on type-conversion failure.
	nnc_tensor* tensor_from_gguf(nnc_context* ctx, gemma_file& gf,
	                             const size_t idx)
	{
		const gguf_file& f = gf.gguf;
		if (idx == SIZE_MAX) return nullptr;
		const auto& gt = f.tensors[idx];

		// K-quant types (Q4_K/Q5_K/Q6_K) are dequanted to BF16 here so
		// the existing BF16 gemv kernels can consume them. Source bytes
		// live in the mmap; the destination buffer is owned by `gf` for
		// the lifetime of the model.
		const bool is_kquant =
			(gt.ggml_type == 12 || gt.ggml_type == 13 || gt.ggml_type == 14);
		if (is_kquant)
		{
			const uint64_t n = gguf_tensor_nelements(gt);
			if (n == 0 || (n % 256) != 0)
			{
				fprintf(stderr,
				        "gemma: K-quant tensor '%s' has %llu elems "
				        "(not multiple of 256)\n",
				        gt.name.c_str(),
				        static_cast<unsigned long long>(n));
				return nullptr;
			}
			const void* src = gguf_tensor_data(f, idx);
			if (!src)
			{
				fprintf(stderr,
				        "gemma: K-quant tensor '%s' has no mapped data\n",
				        gt.name.c_str());
				return nullptr;
			}

			// Dequant Q*_K -> F32 -> BF16. Two-pass to avoid carrying a
			// dedicated K-quant->BF16 kernel; happens once at load time.
			std::vector<float> tmp(static_cast<size_t>(n));
			if (!nnc_dequantize_kquant_to_f32(gt.ggml_type, src, tmp.data(),
			                                  static_cast<size_t>(n)))
			{
				fprintf(stderr, "gemma: dequant failed for '%s'\n",
				        gt.name.c_str());
				return nullptr;
			}
			const size_t total = static_cast<size_t>(n) * sizeof(nnc_bf16_t);
			auto buf = std::unique_ptr<uint8_t[]>(
				new(std::nothrow) uint8_t[total]);
			if (!buf)
			{
				fprintf(stderr,
				        "gemma: alloc %zu bytes failed for dequanted '%s'\n",
				        total, gt.name.c_str());
				return nullptr;
			}
			auto* dst = reinterpret_cast<nnc_bf16_t*>(buf.get());
			for (size_t i = 0; i < static_cast<size_t>(n); ++i)
				dst[i] = nnc_f32_to_bf16(tmp[i]);

			auto* x = nnc_new_tensor_1d(ctx, NNC_TYPE_BF16, 1);
			x->n_dims = static_cast<int>(gt.n_dims);
			for (int d = 0; d < NNC_MAX_DIMS; ++d)
				x->ne[d] = (d < static_cast<int>(gt.n_dims))
					           ? static_cast<int>(gt.ne[d])
					           : 1;
			x->nb[0] = nnc_type_size(NNC_TYPE_BF16);
			x->nb[1] = x->nb[0] * x->ne[0];
			x->nb[2] = x->nb[1] * x->ne[1];
			x->nb[3] = x->nb[2] * x->ne[2];
			x->data = buf.get();
			gf.dequant_buffers.push_back(std::move(buf));
			return x;
		}

		const nnc_type t = to_nnc_type(gt.ggml_type);
		if (t == NNC_TYPE_COUNT)
		{
			fprintf(stderr, "gemma: tensor '%s' has unsupported ggml_type=%u\n",
			        gt.name.c_str(), gt.ggml_type);
			return nullptr;
		}

		// Allocate descriptor in the arena (NOT the data).
		auto* x = nnc_new_tensor_1d(ctx, t, 1);
		x->n_dims = static_cast<int>(gt.n_dims);
		for (int d = 0; d < NNC_MAX_DIMS; ++d)
			x->ne[d] = (d < static_cast<int>(gt.n_dims))
				           ? static_cast<int>(gt.ne[d])
				           : 1;
		x->nb[0] = nnc_type_size(t);
		x->nb[1] = x->nb[0] * x->ne[0];
		x->nb[2] = x->nb[1] * x->ne[1];
		x->nb[3] = x->nb[2] * x->ne[2];
		x->data = const_cast<void*>(gguf_tensor_data(f, idx));
		if (!x->data)
		{
			fprintf(stderr, "gemma: tensor '%s' has no mapped data (offset OOB?)\n",
			        gt.name.c_str());
			return nullptr;
		}
		return x;
	}

	nnc_tensor* tensor_by_name(nnc_context* ctx, gemma_file& gf,
	                           const std::string& name)
	{
		const size_t idx = gguf_find_tensor(gf.gguf, name);
		if (idx == SIZE_MAX) return nullptr;
		return tensor_from_gguf(ctx, gf, idx);
	}

	// Per-tensor weight gemv dispatch. Tensor stores ne[0]=cols (fast dim,
	// the input length to the dot) and ne[1]=rows (number of dots / output
	// length). When `T` is Q8_0 we pull qs and scales from the same
	// allocation (qs first, scales after rows*cols bytes).
	void gw_gemv(const nnc_tensor* T, const float* x, float* y)
	{
		const uint32_t cols = static_cast<uint32_t>(T->ne[0]);
		const uint32_t rows = static_cast<uint32_t>(T->ne[1]);
		if (T->type == NNC_TYPE_Q8_0)
		{
			const auto* qs = static_cast<const int8_t*>(T->data);
			const auto* scales = reinterpret_cast<const float*>(
				qs + static_cast<size_t>(rows) * cols);
			nnc_gemv_q8_0_f32x(qs, scales, x, y, rows, cols);
			return;
		}
		nnc_gemv_bf16w_f32x(T->data, x, y, rows, cols);
	}

	// Per-tensor weight gemv + argmax (lm_head greedy decode). For BF16
	// uses the fused streaming kernel; for Q8_0 materialises the full
	// logits and does a scalar argmax (~1 MB write at vocab=262144,
	// negligible vs the gemv itself).
	int gw_argmax(const nnc_tensor* T, const float* x)
	{
		const uint32_t cols = static_cast<uint32_t>(T->ne[0]);
		const uint32_t rows = static_cast<uint32_t>(T->ne[1]);
		if (T->type == NNC_TYPE_Q8_0)
		{
			std::vector<float> logits(rows);
			const auto* qs = static_cast<const int8_t*>(T->data);
			const auto* scales = reinterpret_cast<const float*>(
				qs + static_cast<size_t>(rows) * cols);
			nnc_gemv_q8_0_f32x(qs, scales, x, logits.data(), rows, cols);
			int best = 0;
			float bv = logits[0];
			for (uint32_t r = 1; r < rows; ++r)
				if (logits[r] > bv)
				{
					bv = logits[r];
					best = static_cast<int>(r);
				}
			return best;
		}
		return nnc_gemv_bf16w_argmax_f32x(T->data, x, rows, cols);
	}
}

bool gemma_load(const std::string& path, gemma_file& out)
{
	if (!gguf_mmap(path, out.gguf)) return false;
	const gguf_file& f = out.gguf;

	// --- architecture sanity check ----------------------------------
	// Accept the gemma3 family in addition to gemma4 — gemma3 is the
	// same tensor topology minus PLE and with a few hparam renames.
	if (!kv_str(f, "general.architecture", out.hparams.arch))
	{
		fprintf(stderr, "gemma: '%s' has no general.architecture\n", path.c_str());
		return false;
	}
	const std::string& arch = out.hparams.arch;
	if (arch != "gemma4" && arch != "gemma3" && arch != "llama")
	{
		fprintf(stderr, "gemma: '%s' has architecture '%s' (expected 'gemma3', 'gemma4', or 'llama')\n",
		        path.c_str(), arch.c_str());
		return false;
	}

	// Helper: build "<arch>.<suffix>" key for this model.
	auto akey = [&](const char* suffix) -> std::string
	{
		return arch + "." + suffix;
	};

	// --- hparams ----------------------------------------------------
	auto& h = out.hparams;
	bool ok = true;
	ok &= kv_int(f, akey("context_length").c_str(), h.n_ctx);
	ok &= kv_int(f, akey("embedding_length").c_str(), h.n_embd);
	ok &= kv_int(f, akey("block_count").c_str(), h.n_layer);
	ok &= kv_int(f, akey("attention.head_count").c_str(), h.n_head);
	ok &= kv_int(f, akey("attention.head_count_kv").c_str(), h.n_head_kv);
	// llama doesn't publish key_length; derive from rope.dimension_count or n_embd/n_head.
	if (!kv_int(f, akey("attention.key_length").c_str(), h.head_dim))
	{
		int rd = 0;
		if (kv_int(f, akey("rope.dimension_count").c_str(), rd) && rd > 0)
			h.head_dim = rd;
		else if (h.n_head > 0)
			h.head_dim = h.n_embd / h.n_head;
	}
	ok &= kv_float(f, akey("rope.freq_base").c_str(), h.rope_freq_base);
	ok &= kv_float(f, akey("attention.layer_norm_rms_epsilon").c_str(), h.rms_eps);
	if (!ok)
	{
		fprintf(stderr, "gemma: missing required hparam in '%s'\n", path.c_str());
		return false;
	}

	// Optional gemma4-only fields (defaulted otherwise).
	kv_int(f, akey("attention.key_length_swa").c_str(), h.head_dim_swa);
	if (h.head_dim_swa <= 0) h.head_dim_swa = h.head_dim;
	kv_int(f, akey("rope.dimension_count").c_str(), h.rope_dim);
	if (h.rope_dim <= 0) h.rope_dim = h.head_dim; // gemma3 omits this
	kv_int(f, akey("rope.dimension_count_swa").c_str(), h.rope_dim_swa);
	if (h.rope_dim_swa <= 0) h.rope_dim_swa = h.rope_dim;
	kv_float(f, akey("rope.freq_base_swa").c_str(), h.rope_freq_base_swa);
	if (h.rope_freq_base_swa <= 0)
	{
		// gemma3 uses a hardcoded local-rope theta of 10000 for sliding layers
		// (HF: rope_local_base_freq); not stored in GGUF.
		if (h.arch == "gemma3")
			h.rope_freq_base_swa = 10000.0f;
		else
			h.rope_freq_base_swa = h.rope_freq_base;
	}
	kv_int(f, akey("attention.sliding_window").c_str(), h.sliding_window);
	kv_int(f, akey("attention.shared_kv_layers").c_str(), h.shared_kv_layers);
	kv_int(f, akey("embedding_length_per_layer_input").c_str(), h.ple_dim);
	kv_float(f, akey("final_logit_softcapping").c_str(), h.final_logit_softcap);
	bool have_attn_scale = kv_float(f, akey("attention.scale").c_str(), h.attention_scale);
	if (!have_attn_scale)
	{
		// gemma3/llama have no q_norm-absorbed scale; gemma4 (3n) does.
		if ((h.arch == "gemma3" || h.arch == "llama") && h.head_dim > 0)
			h.attention_scale = 1.0f / std::sqrt(static_cast<float>(h.head_dim));
		else
			h.attention_scale = 1.0f;
	}
	if (h.attention_scale <= 0) h.attention_scale = 1.0f;

	// RoPE linear scaling: theta_i is scaled by `1 / factor` (i.e.
	// effective_pos = pos / factor). Both gemma3-4B and recent llama
	// finetunes use this. Read string + float from the GGUF metadata.
	{
		std::string scaling_type;
		(void)kv_str(f, akey("rope.scaling.type").c_str(), scaling_type);
		float factor = 0.0f;
		(void)kv_float(f, akey("rope.scaling.factor").c_str(), factor);
		if (scaling_type == "linear" && factor > 0)
			h.rope_freq_scale = 1.0f / factor;
	}

	kv_int(f, "general.file_type", h.file_type);
	if (!ok)
	{
		fprintf(stderr, "gemma: missing required hparam in '%s'\n", path.c_str());
		return false;
	}

	// --- vocab ------------------------------------------------------
	const gguf_value* tokens = gguf_find_kv(f, "tokenizer.ggml.tokens");
	if (!tokens || tokens->type != GGUF_TYPE_ARRAY || tokens->arr_type != GGUF_TYPE_STRING)
	{
		fprintf(stderr, "gemma: missing tokenizer.ggml.tokens\n");
		return false;
	}
	out.vocab_tokens.resize(tokens->arr.size());
	for (size_t i = 0; i < tokens->arr.size(); ++i)
		out.vocab_tokens[i] = tokens->arr[i].str;
	h.n_vocab = static_cast<int>(out.vocab_tokens.size());

	if (const gguf_value* sc = gguf_find_kv(f, "tokenizer.ggml.scores");
		sc && sc->type == GGUF_TYPE_ARRAY)
	{
		out.vocab_scores.resize(sc->arr.size());
		for (size_t i = 0; i < sc->arr.size(); ++i)
			out.vocab_scores[i] = static_cast<float>(sc->arr[i].f64);
	}
	if (const gguf_value* tt = gguf_find_kv(f, "tokenizer.ggml.token_type");
		tt && tt->type == GGUF_TYPE_ARRAY)
	{
		out.vocab_token_types.resize(tt->arr.size());
		for (size_t i = 0; i < tt->arr.size(); ++i)
			out.vocab_token_types[i] = static_cast<int32_t>(tt->arr[i].i64);
	}
	int v = 0;
	if (kv_int(f, "tokenizer.ggml.bos_token_id", v)) out.bos_id = v;
	if (kv_int(f, "tokenizer.ggml.eos_token_id", v)) out.eos_id = v;
	if (kv_int(f, "tokenizer.ggml.unknown_token_id", v)) out.unk_id = v;
	if (kv_int(f, "tokenizer.ggml.padding_token_id", v)) out.pad_id = v;

	// add_bos_token / add_space_prefix flags (booleans).
	if (const gguf_value* b = gguf_find_kv(f, "tokenizer.ggml.add_bos_token");
		b && b->type == GGUF_TYPE_BOOL)
		out.add_bos_token = (b->u64 != 0);
	if (const gguf_value* b = gguf_find_kv(f, "tokenizer.ggml.add_space_prefix");
		b && b->type == GGUF_TYPE_BOOL)
		out.add_space_prefix = (b->u64 != 0);

	// merges: array<string>. Each entry is "lhs<space>rhs"; rank = index.
	if (const gguf_value* mg = gguf_find_kv(f, "tokenizer.ggml.merges");
		mg && mg->type == GGUF_TYPE_ARRAY && mg->arr_type == GGUF_TYPE_STRING)
	{
		out.merges.resize(mg->arr.size());
		out.merge_rank.reserve(mg->arr.size() * 2);
		for (size_t i = 0; i < mg->arr.size(); ++i)
		{
			out.merges[i] = mg->arr[i].str;
			out.merge_rank.emplace(out.merges[i], static_cast<int>(i));
		}
	}

	// token_to_id reverse lookup.
	out.token_to_id.reserve(out.vocab_tokens.size() * 2);
	for (size_t i = 0; i < out.vocab_tokens.size(); ++i)
	{
		out.token_to_id.emplace(out.vocab_tokens[i], static_cast<int>(i));
		if (out.vocab_tokens[i].size() > out.max_token_bytes)
			out.max_token_bytes = out.vocab_tokens[i].size();
	}

	// tokenizer model name (selects BPE vs Unigram in gemma_tokenize).
	if (const gguf_value* tm = gguf_find_kv(f, "tokenizer.ggml.model");
		tm && tm->type == GGUF_TYPE_STRING)
		out.tokenizer_model = tm->str;

	// --- arena for tensor descriptors -------------------------------
	// Each descriptor is ~80 bytes; 700 tensors * 80 + slack = a few hundred KB.
	constexpr size_t mem = 4ull * 1024 * 1024;
	constexpr nnc_init_params ip{mem, nullptr};
	out.ctx = nnc_init(ip);
	if (!out.ctx)
	{
		fprintf(stderr, "gemma: nnc_init failed\n");
		return false;
	}

	// --- top-level weights ------------------------------------------
	out.token_embd = tensor_by_name(out.ctx, out, "token_embd.weight");
	out.per_layer_token_embd = tensor_by_name(out.ctx, out, "per_layer_token_embd.weight");
	out.output_norm = tensor_by_name(out.ctx, out, "output_norm.weight");
	out.output = tensor_by_name(out.ctx, out, "output.weight");
	out.rope_freqs = tensor_by_name(out.ctx, out, "rope_freqs.weight");
	out.per_layer_model_proj = tensor_by_name(out.ctx, out, "per_layer_model_proj.weight");
	out.per_layer_proj_norm = tensor_by_name(out.ctx, out, "per_layer_proj_norm.weight");
	if (!out.token_embd)
	{
		fprintf(stderr, "gemma: missing token_embd.weight\n");
		return false;
	}

	// --- per-layer weights ------------------------------------------
	out.layers.resize(h.n_layer);
	const gguf_value* swp = gguf_find_kv(f, akey("attention.sliding_window_pattern").c_str());
	const gguf_value* ffl = gguf_find_kv(f, akey("feed_forward_length").c_str());

	// gemma3 omits the per-layer pattern array but uses an int periodicity
	// where every Nth layer is a global (non-SWA) layer. The standard
	// gemma3 release uses a period of 6.
	int swa_period = 0;
	if (arch == "gemma3" && h.sliding_window > 0)
		swa_period = 6;

	for (int li = 0; li < h.n_layer; ++li)
	{
		auto& L = out.layers[li];
		char buf[64];

		// E2B has feed_forward_length as an array; E4B has a scalar.
		if (ffl && ffl->type == GGUF_TYPE_ARRAY && ffl->arr.size() == static_cast<size_t>(h.n_layer))
			L.n_ff = static_cast<int>(ffl->arr[li].i64);
		else if (ffl)
			L.n_ff = static_cast<int>(ffl->u64);

		if (swp && swp->type == GGUF_TYPE_ARRAY && swp->arr.size() == static_cast<size_t>(h.n_layer))
			L.sliding_window = (swp->arr[li].u64 != 0);
		else if (swa_period > 0)
		{
			// gemma3: every (li+1)%period == 0 layer is a global (full-context)
			// layer; the rest use SWA. Matches Hugging Face's reference
			// implementation `is_sliding = bool((i + 1) % sliding_window_pattern)`.
			L.sliding_window = (((li + 1) % swa_period) != 0);
		}

#define G(field, name) do { \
	snprintf(buf, sizeof(buf), "blk.%d." name, li); \
	L.field = tensor_by_name(out.ctx, out, buf); \
} while (0)

		G(attn_norm, "attn_norm.weight");
		G(ffn_norm, "ffn_norm.weight");
		G(post_attention_norm, "post_attention_norm.weight");
		G(post_ffw_norm, "post_ffw_norm.weight");
		G(post_norm, "post_norm.weight");
		G(attn_q_norm, "attn_q_norm.weight");
		G(attn_k_norm, "attn_k_norm.weight");
		G(attn_q, "attn_q.weight");
		G(attn_k, "attn_k.weight");
		G(attn_v, "attn_v.weight");
		G(attn_output, "attn_output.weight");
		G(ffn_gate, "ffn_gate.weight");
		G(ffn_up, "ffn_up.weight");
		G(ffn_down, "ffn_down.weight");
		G(inp_gate, "inp_gate.weight");
		G(proj, "proj.weight");
		G(layer_output_scale, "layer_output_scale.weight");
#undef G
	}
	return true;
}

void gemma_free(gemma_file& f)
{
	if (f.ctx)
	{
		nnc_free(f.ctx);
		f.ctx = nullptr;
	}
	gguf_unmap(f.gguf);
	f.layers.clear();
	f.vocab_tokens.clear();
	f.vocab_scores.clear();
	f.vocab_token_types.clear();
}

void gemma_print_info(const gemma_file& f)
{
	const auto& h = f.hparams;
	printf("gemma: arch=%s file_type=%d\n", h.arch.c_str(), h.file_type);
	printf("  n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d (swa=%d)\n",
	       h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim, h.head_dim_swa);
	printf("  rope_dim=%d (swa=%d) freq_base=%g (swa=%g) rms_eps=%g\n",
	       h.rope_dim, h.rope_dim_swa, h.rope_freq_base, h.rope_freq_base_swa, h.rms_eps);
	printf("  ctx=%d sliding_window=%d shared_kv_layers=%d ple_dim=%d softcap=%g\n",
	       h.n_ctx, h.sliding_window, h.shared_kv_layers, h.ple_dim, h.final_logit_softcap);
	printf("  attention_scale=%g rope_freq_scale=%g\n",
	       h.attention_scale, h.rope_freq_scale);
	printf("  vocab=%d tokens (bos=%d eos=%d unk=%d pad=%d)\n",
	       h.n_vocab, f.bos_id, f.eos_id, f.unk_id, f.pad_id);

	auto count_present = [](const gemma_file& f, nnc_tensor* (gemma_layer::*p))
	{
		int n = 0;
		for (const auto& L : f.layers) if (L.*p) ++n;
		return n;
	};
	auto report = [&](const char* name, nnc_tensor* gemma_layer::* p)
	{
		const int n = count_present(f, p);
		const char* tag = (n == h.n_layer) ? "ok" : (n == 0 ? "MISSING" : "partial");
		printf("  layer.%-22s : %3d / %d  [%s]\n", name, n, h.n_layer, tag);
	};
	printf("\nper-layer tensor presence:\n");
	report("attn_norm", &gemma_layer::attn_norm);
	report("ffn_norm", &gemma_layer::ffn_norm);
	report("post_attention_norm", &gemma_layer::post_attention_norm);
	report("post_ffw_norm", &gemma_layer::post_ffw_norm);
	report("post_norm", &gemma_layer::post_norm);
	report("attn_q_norm", &gemma_layer::attn_q_norm);
	report("attn_k_norm", &gemma_layer::attn_k_norm);
	report("attn_q", &gemma_layer::attn_q);
	report("attn_k", &gemma_layer::attn_k);
	report("attn_v", &gemma_layer::attn_v);
	report("attn_output", &gemma_layer::attn_output);
	report("ffn_gate", &gemma_layer::ffn_gate);
	report("ffn_up", &gemma_layer::ffn_up);
	report("ffn_down", &gemma_layer::ffn_down);
	report("inp_gate", &gemma_layer::inp_gate);
	report("proj", &gemma_layer::proj);
	report("layer_output_scale", &gemma_layer::layer_output_scale);

	printf("\ntop-level: token_embd=%s per_layer_token_embd=%s output_norm=%s rope_freqs=%s\n",
	       f.token_embd ? "ok" : "MISSING",
	       f.per_layer_token_embd ? "ok" : "MISSING",
	       f.output_norm ? "ok" : "MISSING",
	       f.rope_freqs ? "ok" : "MISSING");

	// Sliding-window pattern summary.
	int sw = 0;
	for (const auto& L : f.layers) if (L.sliding_window) ++sw;
	printf("sliding-window layers: %d / %d\n", sw, h.n_layer);

	// Per-layer FFN sizes.
	int min_ff = INT32_MAX, max_ff = 0;
	for (const auto& L : f.layers)
	{
		if (L.n_ff < min_ff) min_ff = L.n_ff;
		if (L.n_ff > max_ff) max_ff = L.n_ff;
	}
	printf("per-layer n_ff: min=%d max=%d\n", min_ff, max_ff);
}

// ----------------------------------------------------------------------------
// Probe: partial forward pass on a single token through layer 0.
// ----------------------------------------------------------------------------

namespace
{
	struct stats
	{
		float lo, hi;
		double mean, rms;
	};

	stats compute_stats(const float* p, const size_t n)
	{
		stats s{
			std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), 0.0, 0.0
		};
		double sum = 0.0, sum2 = 0.0;
		for (size_t i = 0; i < n; ++i)
		{
			const float v = p[i];
			if (v < s.lo) s.lo = v;
			if (v > s.hi) s.hi = v;
			sum += v;
			sum2 += static_cast<double>(v) * v;
		}
		s.mean = n ? sum / static_cast<double>(n) : 0.0;
		s.rms = n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0;
		return s;
	}

	void print_stats(const char* tag, const float* p, const size_t n)
	{
		const stats s = compute_stats(p, n);
		printf("  %-26s n=%-6zu min=%+.4g max=%+.4g mean=%+.4g rms=%.4g\n",
		       tag, n, static_cast<double>(s.lo), static_cast<double>(s.hi),
		       s.mean, s.rms);
	}
}

int gemma_probe(const gemma_file& f, const int token_id)
{
	const auto& h = f.hparams;
	if (token_id < 0 || token_id >= h.n_vocab)
	{
		fprintf(stderr, "gemma_probe: token_id %d out of range [0,%d)\n",
		        token_id, h.n_vocab);
		return 1;
	}
	if (f.layers.empty() || !f.token_embd || !f.layers[0].attn_norm
		|| !f.layers[0].attn_q || !f.layers[0].attn_q_norm)
	{
		fprintf(stderr, "gemma_probe: required tensors missing\n");
		return 1;
	}

	const gemma_layer& L0 = f.layers[0];
	const int n_embd = h.n_embd;
	const int q_dim = L0.attn_q->ne[1]; // [n_embd, q_dim]
	// Derive head_dim from the per-head Q-norm tensor shape (see layer_forward).
	const int head_dim = L0.attn_q_norm->ne[0];
	const int n_head = q_dim / head_dim;

	printf("gemma_probe: token_id=%d ('%s')\n", token_id,
	       (token_id < static_cast<int>(f.vocab_tokens.size())) ? f.vocab_tokens[token_id].c_str() : "?");
	printf("  layer 0: sliding_window=%d  q_dim=%d head_dim=%d n_head=%d\n",
	       L0.sliding_window ? 1 : 0, q_dim, head_dim, n_head);
	if (L0.layer_output_scale)
	{
		const auto lo = static_cast<const float*>(L0.layer_output_scale->data);
		const long long n0 = L0.layer_output_scale->ne[0];
		printf("  layer_output_scale: ne=[%lld,%lld,%lld,%lld]  v[0]=%g v[1]=%g v[last]=%g\n",
		       n0,
		       static_cast<long long>(L0.layer_output_scale->ne[1]),
		       static_cast<long long>(L0.layer_output_scale->ne[2]),
		       static_cast<long long>(L0.layer_output_scale->ne[3]),
		       (n0 > 0) ? lo[0] : 0.0f,
		       (n0 > 1) ? lo[1] : 0.0f,
		       (n0 > 0) ? lo[n0 - 1] : 0.0f);
	}
	if (L0.post_norm)
	{
		printf("  post_norm:          ne=[%lld,%lld,%lld,%lld]\n",
		       static_cast<long long>(L0.post_norm->ne[0]), static_cast<long long>(L0.post_norm->ne[1]),
		       static_cast<long long>(L0.post_norm->ne[2]), static_cast<long long>(L0.post_norm->ne[3]));
	}

	// 1. Token embedding * sqrt(n_embd) (Gemma scales embeddings).
	std::vector<float> x(n_embd);
	const float embd_scale = std::sqrt(static_cast<float>(n_embd));
	nnc_embed_row_bf16(x.data(), f.token_embd->data, token_id,
	                   static_cast<size_t>(n_embd), embd_scale);
	print_stats("embed*sqrt(n_embd)", x.data(), n_embd);

	// 2. RMSNorm.
	std::vector<float> norm(n_embd);
	nnc_rmsnorm_f32(norm.data(), x.data(), n_embd, h.rms_eps);
	print_stats("rmsnorm(embed)", norm.data(), n_embd);

	// 3. Apply attn_norm gamma. Gemma4 uses plain RMSNorm + gamma multiply
	// (norm_shift = 0.0 in convert_hf_to_gguf.py: weights stored as-is).
	std::vector<float> normed(n_embd);
	const auto gamma = static_cast<const float*>(L0.attn_norm->data);
	for (int i = 0; i < n_embd; ++i)
		normed[i] = norm[i] * gamma[i];
	print_stats("normed * gamma", normed.data(), n_embd);

	// 4. Q projection: y[r] = sum_k W[r,k] * normed[k]; W is BF16
	//    [n_embd, q_dim] in column-major-by-output convention so the
	//    fast dim is the input. nnc_gemv_bf16w_f32x reads rows as
	//    [rows, cols] = [q_dim, n_embd].
	std::vector<float> q(q_dim);
	gw_gemv(L0.attn_q, normed.data(), q.data());
	print_stats("Q = W_q * normed", q.data(), q_dim);

	// 5. attn_q_norm is per-head_dim (RMSNorm + (1+gamma) again).
	const auto q_gamma = static_cast<const float*>(L0.attn_q_norm->data);
	for (int hh = 0; hh < n_head; ++hh)
	{
		float* qh = q.data() + hh * head_dim;
		nnc_rmsnorm_f32(qh, qh, head_dim, h.rms_eps);
		for (int i = 0; i < head_dim; ++i) qh[i] *= q_gamma[i];
	}
	print_stats("Q after q_norm", q.data(), q_dim);

	// 6. RoPE on Q (NeoX-style; n_rot lanes of each head's head_dim).
	int n_rot = L0.sliding_window ? h.rope_dim_swa : h.rope_dim;
	if (n_rot > head_dim) n_rot = head_dim;
	const float freq = L0.sliding_window ? h.rope_freq_base_swa : h.rope_freq_base;
	constexpr int pos = 0;
	nnc_rope_f32(q.data(), static_cast<uint32_t>(n_head),
	             static_cast<uint32_t>(head_dim),
	             static_cast<uint32_t>(n_rot), pos, freq);
	print_stats("Q after RoPE (pos=0)", q.data(), q_dim);

	// --- full forward sweep: run all layers via gemma_eval_token, log RMS per layer ---
	{
		printf("\n--- full forward sweep (greedy, pos=0) ---\n");
		gemma_kv_cache cache;
		if (!gemma_kv_init(f, cache, 32))
		{
			printf("kv_init failed\n");
			return 0;
		}

		// Build a one-shot eval: token = bos (token_id=2). We re-use gemma_eval_token
		// but instrument by running it twice: once normal, then a manual layer-by-layer
		// trace for the first 35 layers using a private mini-loop. Simpler: just call
		// gemma_eval_token and print top-3 logits.
		std::vector<float> logits(h.n_vocab);
		const int rc = gemma_eval_token(f, cache, token_id, /*pos*/ 0, logits.data());
		if (rc != 0)
		{
			printf("eval failed rc=%d\n", rc);
			gemma_kv_free(cache);
			return 0;
		}
		float lo = logits[0], hi = logits[0];
		double sum = 0.0;
		int argmax = 0;
		for (int i = 0; i < h.n_vocab; ++i)
		{
			if (logits[i] < lo) lo = logits[i];
			if (logits[i] > hi)
			{
				hi = logits[i];
				argmax = i;
			}
			sum += logits[i];
		}
		printf("logits: min=%g max=%g mean=%g argmax=%d ('%s')\n",
		       lo, hi, sum / h.n_vocab, argmax,
		       (argmax < static_cast<int>(f.vocab_tokens.size())) ? f.vocab_tokens[argmax].c_str() : "?");

		// Top-5 logits
		std::vector<int> idx(h.n_vocab);
		for (int i = 0; i < h.n_vocab; ++i) idx[i] = i;
		std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
		                  [&](const int a, const int b) { return logits[a] > logits[b]; });
		printf("top-5:");
		for (int k = 0; k < 5; ++k)
			printf("  [%d %g '%s']", idx[k], logits[idx[k]],
			       (idx[k] < static_cast<int>(f.vocab_tokens.size())) ? f.vocab_tokens[idx[k]].c_str() : "?");
		printf("\n");

		gemma_kv_free(cache);
	}

	return 0;
}

// ----------------------------------------------------------------------------
// Full single-token forward pass (pos=0). PLE / shared-KV / sliding mask
// are deliberately stubbed; this is a numerical smoke test, not a faithful
// generator. With a single position, the causal softmax over one key is
// trivially [1.0], so the attention output equals V projected by W_O.
// ----------------------------------------------------------------------------

namespace
{
	// Apply gamma * x in place (Gemma4 RMSNorm convention; norm_shift=0).
	void apply_gamma(float* x, const float* gamma_stored, const size_t n)
	{
		for (size_t i = 0; i < n; ++i) x[i] *= gamma_stored[i];
	}

	void rmsnorm_with_gamma(float* y, const float* x, const size_t n,
	                        const float* gamma_stored, const float eps)
	{
		nnc_rmsnorm_f32(y, x, n, eps);
		apply_gamma(y, gamma_stored, n);
	}

	// Plain F32 dot product (used for per-head attention scores). Routes
	// to the AVX2 4-accumulator helper from nn_ops; for head_dim 256 / 512
	// the call overhead is amortised by the unrolled inner loop.
	float dot_f32(const float* a, const float* b, const int n)
	{
		return nnc_dot_f32_simd(a, b, static_cast<size_t>(n));
	}

	struct layer_scratch
	{
		float* normed; // [n_embd]
		float* q; // [max_q_dim]
		float* k_cur; // [max_kv_dim]   current-position K (pre-cache)
		float* v_cur; // [max_kv_dim]   current-position V (pre-cache)
		float* attn_out; // [max_q_dim]    per-head attention output
		float* gate; // [max_ff]
		float* up; // [max_ff]
		float* scores; // [n_ctx]        per-head softmax scratch
		float* ple_gate; // [ple_dim]      PLE inp_gate output
		float* ple_contrib; // [n_embd]       PLE projection back to residual
		float* ple_for_layer; // [ple_dim]      pointer into per-token PLE table for this layer
	};

	void layer_forward(const gemma_file& f, gemma_kv_cache& cache, const int li,
	                   const int pos, std::vector<float>& x, const layer_scratch& s)
	{
		const auto& h = f.hparams;
		const gemma_layer& L = f.layers[li];
		const int n_embd = h.n_embd;
		const int q_dim = L.attn_q->ne[1];
		const int kv_dim = L.attn_k->ne[1];
		// gemma4: derive head_dim from the per-head Q-norm tensor shape.
		// The hparam key_length=512 disagrees with the actual tensor shapes
		// (q_dim=2048, k_dim=256, q_norm=[256]) — all layers use head_dim=256
		// regardless of the sliding-window flag.
		// llama / gemma3 don't have q_norm; fall back to hparams.
		const int head_dim = L.attn_q_norm
			? L.attn_q_norm->ne[0]
			: h.head_dim;
		const int n_head = q_dim / head_dim;
		const int n_kv = kv_dim / head_dim; // 1 on E2B
		int n_rot = L.sliding_window ? h.rope_dim_swa : h.rope_dim;
		if (n_rot > head_dim) n_rot = head_dim;
		const float freq = L.sliding_window ? h.rope_freq_base_swa : h.rope_freq_base;

		// --- attention block ---
		float* normed = s.normed;
		rmsnorm_with_gamma(normed, x.data(), n_embd,
		                   static_cast<const float*>(L.attn_norm->data), h.rms_eps);

		float* q = s.q;
		float* k = s.k_cur;
		float* v = s.v_cur;

		// Determine if this layer reuses an earlier layer's KV (gemma4 shared_kv_layers).
		// llama.cpp semantics: GGUF key 'attention.shared_kv_layers' is the COUNT of
		// layers that REUSE KV; n_layer_kv_from_start = n_layer - count is the threshold.
		// For E2B: count=20, n_layer=35 -> threshold=15. Layers 15..34 reuse KV from
		// layer 14 (full-attn) or 13 (sliding) of layers 0..14.
		const int kv_from_start = (h.shared_kv_layers > 0)
			                          ? (h.n_layer - h.shared_kv_layers)
			                          : h.n_layer;
		const bool reuse_kv = (li >= kv_from_start);
		const int li_kv = reuse_kv
			                  ? (kv_from_start - (L.sliding_window ? 2 : 1))
			                  : li;
		// The shared-KV math above assumes the source layer's
		// sliding-window flag matches this layer's, i.e. the alternation
		// pattern stays consistent across the reuse boundary. Verify it
		// (cheap, fires loudly if a future GGUF breaks the assumption).
		if (reuse_kv)
		{
			NNC_ASSERT(li_kv >= 0 && li_kv < h.n_layer);
			NNC_ASSERT(f.layers[li_kv].sliding_window == L.sliding_window
				&& "shared-KV pattern mismatch: source layer's SWA flag differs");
		}

		gw_gemv(L.attn_q, normed, q);
		if (!reuse_kv)
		{
			gw_gemv(L.attn_k, normed, k);
			gw_gemv(L.attn_v, normed, v);
		}

		// Per-head Q / K norms (gemma4 only; gemma3 / llama lack these).
		if (L.attn_q_norm)
		{
			const auto qg = static_cast<const float*>(L.attn_q_norm->data);
			nnc_rmsnorm_gamma_multi_f32(q, q, n_head, head_dim, qg, h.rms_eps);
		}

		if (!reuse_kv)
		{
			if (L.attn_k_norm)
			{
				const auto kg = static_cast<const float*>(L.attn_k_norm->data);
				nnc_rmsnorm_gamma_multi_f32(k, k, n_kv, head_dim, kg, h.rms_eps);
			}

			// Per-head V plain RMSNorm (no learned gamma) — gemma4 specific.
			// gemma3 / llama do NOT normalize V; skip there.
			if (h.arch == "gemma4")
			{
				for (int hd = 0; hd < n_kv; ++hd)
				{
					float* vh = v + hd * head_dim;
					float ss = 0.0f;
					for (int i = 0; i < head_dim; ++i) ss += vh[i] * vh[i];
					const float inv = 1.0f / std::sqrt(ss / head_dim + h.rms_eps);
					for (int i = 0; i < head_dim; ++i) vh[i] *= inv;
				}
			}
		}

		// RoPE on Q always; K only when freshly computed (cached K is already rotated).
		// Linear position scaling applies only to global (non-SWA) layers
		// in gemma3 (HF: rope_scaling only on rope_theta, not rope_local_base_freq).
		// gemma4 applies the same scale across all layers.
		const float layer_rope_scale = (h.arch == "gemma3" && L.sliding_window)
			                               ? 1.0f
			                               : h.rope_freq_scale;
		const float rpos = static_cast<float>(pos) * layer_rope_scale;
		nnc_rope_f32(q, n_head, head_dim, n_rot, rpos, freq);
		if (!reuse_kv)
			nnc_rope_f32(k, n_kv, head_dim, n_rot, rpos, freq);

		// Append current K/V to per-layer cache (only when freshly computed).
		float* Kbase = cache.k[li_kv].data();
		float* Vbase = cache.v[li_kv].data();
		if (!reuse_kv)
		{
			std::memcpy(Kbase + pos * kv_dim, k, sizeof(float) * kv_dim);
			std::memcpy(Vbase + pos * kv_dim, v, sizeof(float) * kv_dim);
		}

		// Attention window (causal + optional sliding):
		const int win = L.sliding_window ? h.sliding_window : (pos + 1);
		const int t_start = (pos + 1 > win) ? (pos + 1 - win) : 0;
		const int t_end = pos + 1; // inclusive of current pos
		const int n_t = t_end - t_start;
		// Pre-softmax attention scale. Gemma 3n GGUFs typically omit
		// this (defaults to 1.0); allow override via hparams in case a
		// future variant publishes a real value.
		const float scale = (h.attention_scale > 0) ? h.attention_scale : 1.0f;

		float* attn = s.attn_out;
		float* scores = s.scores;
		for (int hd = 0; hd < n_head; ++hd)
		{
			const int kv_h = (n_kv == 1) ? 0 : (hd * n_kv / n_head);
			const float* qh = q + hd * head_dim;

			// Scores: dot(Q[h], K[t][kv_h]) / sqrt(d).
			for (int t = 0; t < n_t; ++t)
			{
				const float* kt = Kbase + (t_start + t) * kv_dim + kv_h * head_dim;
				scores[t] = dot_f32(qh, kt, head_dim) * scale;
			}

			// Fused softmax + V matmul: single sweep over t, no
			// memset of `ah`, no second read of `scores`.
			float* ah = attn + hd * head_dim;
			const float* Vh = Vbase + t_start * kv_dim + kv_h * head_dim;
			nnc_attn_softmax_v_f32(ah, scores, Vh,
			                       static_cast<size_t>(n_t),
			                       static_cast<size_t>(kv_dim),
			                       static_cast<size_t>(head_dim));
		}

		// Output projection: [q_dim] -> [n_embd].
		float* attn_out = s.normed; // reuse
		gw_gemv(L.attn_output, attn, attn_out);

		// Post-attention RMSNorm (gemma4) + residual. llama / gemma3 omit this norm.
		if (L.post_attention_norm)
		{
			rmsnorm_with_gamma(attn_out, attn_out, n_embd,
			                   static_cast<const float*>(L.post_attention_norm->data), h.rms_eps);
		}
		nnc_add_inplace_f32(x.data(), attn_out, n_embd);

		// --- MLP block (gated, PAR) ---
		// Gemma uses gelu(gate)*up; llama uses silu(gate)*up (SwiGLU).
		rmsnorm_with_gamma(normed, x.data(), n_embd,
		                   static_cast<const float*>(L.ffn_norm->data), h.rms_eps);

		const int n_ff = L.n_ff;
		float* gate = s.gate;
		float* up = s.up;
		gw_gemv(L.ffn_gate, normed, gate);
		gw_gemv(L.ffn_up, normed, up);
		if (h.arch == "llama")
			nnc_swiglu_f32(gate, gate, up, static_cast<size_t>(n_ff));
		else
			nnc_gelu_mul_f32(gate, gate, up, static_cast<size_t>(n_ff));

		float* ffn_out = s.normed;
		gw_gemv(L.ffn_down, gate, ffn_out);

		// Post-FFW RMSNorm (gemma4) — llama / gemma3 omit.
		if (L.post_ffw_norm)
		{
			rmsnorm_with_gamma(ffn_out, ffn_out, n_embd,
			                   static_cast<const float*>(L.post_ffw_norm->data), h.rms_eps);
		}
		nnc_add_inplace_f32(x.data(), ffn_out, n_embd);

		// --- Per-Layer Embedding (PLE) block ---
		// pe_in = x (current residual); g = gelu(inp_gate(pe_in)) * inp_per_layer[li]
		// contrib = post_norm( proj(g) ); x = pe_in + contrib
		if (L.inp_gate && L.proj && s.ple_for_layer)
		{
			const int ple_dim = L.inp_gate->ne[1]; // [n_embd, ple_dim]
			float* g = s.ple_gate;
			gw_gemv(L.inp_gate, x.data(), g);
			nnc_gelu_f32(g, g, static_cast<size_t>(ple_dim));
			nnc_mul_inplace_f32(g, s.ple_for_layer, static_cast<size_t>(ple_dim));

			float* contrib = s.ple_contrib;
			gw_gemv(L.proj, g, contrib);
			if (L.post_norm)
				rmsnorm_with_gamma(contrib, contrib, n_embd,
				                   static_cast<const float*>(L.post_norm->data), h.rms_eps);
			nnc_add_inplace_f32(x.data(), contrib, n_embd);
		}

		// --- per-layer output scale (gemma4 layer_scalar) ---
		// Multiplies the entire residual stream before next layer.
		if (L.layer_output_scale && L.layer_output_scale->ne[0] > 0)
		{
			const float s_val = static_cast<const float*>(L.layer_output_scale->data)[0];
			for (int i = 0; i < n_embd; ++i) x.data()[i] *= s_val;
		}
	}
}

// ----------------------------------------------------------------------------
// KV cache lifecycle.
// ----------------------------------------------------------------------------

bool gemma_kv_init(const gemma_file& f, gemma_kv_cache& cache, const int n_ctx)
{
	const auto& h = f.hparams;
	const int n_layer = h.n_layer;
	cache.n_ctx = n_ctx;
	cache.cur_pos = 0;
	cache.kv_dim_per_layer.assign(n_layer, 0);
	cache.k.assign(n_layer, {});
	cache.v.assign(n_layer, {});
	size_t total_bytes = 0;
	int max_q_dim = 0, max_kv_dim = 0, max_ff = 0;
	for (int li = 0; li < n_layer; ++li)
	{
		const int kv_dim = f.layers[li].attn_k ? f.layers[li].attn_k->ne[1] : 0;
		cache.kv_dim_per_layer[li] = kv_dim;
		const size_t n = static_cast<size_t>(n_ctx) * static_cast<size_t>(kv_dim);
		cache.k[li].assign(n, 0.0f);
		cache.v[li].assign(n, 0.0f);
		total_bytes += 2 * n * sizeof(float);

		if (f.layers[li].attn_q && f.layers[li].attn_q->ne[1] > max_q_dim)
			max_q_dim = f.layers[li].attn_q->ne[1];
		if (kv_dim > max_kv_dim) max_kv_dim = kv_dim;
		if (f.layers[li].n_ff > max_ff) max_ff = f.layers[li].n_ff;
	}

	// One-time scratch allocation (resize, not assign — first eval_token
	// will write every element before reading it).
	const int ple_dim = (h.ple_dim > 0) ? h.ple_dim : 0;
	const int ple_total = ple_dim * n_layer;
	cache.sx.resize(h.n_embd);
	cache.snormed.resize(h.n_embd);
	cache.sq.resize(max_q_dim);
	cache.sk_cur.resize(max_kv_dim);
	cache.sv_cur.resize(max_kv_dim);
	cache.sattn.resize(max_q_dim);
	cache.sgate.resize(max_ff);
	cache.sup.resize(max_ff);
	cache.sscores.resize(n_ctx);
	cache.sple_gate.resize(ple_dim > 0 ? ple_dim : 1);
	cache.sple_contrib.resize(h.n_embd);
	cache.sple_inputs.resize(ple_total > 0 ? ple_total : 1);
	cache.sple_proj.resize(ple_total > 0 ? ple_total : 1);

	printf("gemma_kv_init: n_ctx=%d  cache size = %.1f MB\n",
	       n_ctx, total_bytes / (1024.0 * 1024.0));
	return true;
}

void gemma_kv_free(gemma_kv_cache& cache)
{
	cache.k.clear();
	cache.v.clear();
	cache.kv_dim_per_layer.clear();
	cache.sx.clear();
	cache.snormed.clear();
	cache.sq.clear();
	cache.sk_cur.clear();
	cache.sv_cur.clear();
	cache.sattn.clear();
	cache.sgate.clear();
	cache.sup.clear();
	cache.sscores.clear();
	cache.sple_gate.clear();
	cache.sple_contrib.clear();
	cache.sple_inputs.clear();
	cache.sple_proj.clear();
	cache.n_ctx = 0;
	cache.cur_pos = 0;
}

// ----------------------------------------------------------------------------
// Single-token eval against an existing KV cache.
// ----------------------------------------------------------------------------

namespace
{
	// Runs every step of the per-token forward pass except the final
	// lm_head projection: embed -> PLE inputs -> all layers -> output_norm.
	// On success, the post-output_norm hidden state lives in cache.sx
	// (length h.n_embd) ready to feed into the lm_head.
	int gemma_forward_to_x(const gemma_file& f, gemma_kv_cache& cache,
	                       const int token_id, const int pos)
	{
		const auto& h = f.hparams;
		if (pos != cache.cur_pos)
		{
			fprintf(stderr, "gemma_eval_token: pos=%d != cache.cur_pos=%d\n",
			        pos, cache.cur_pos);
			return 1;
		}
		if (pos >= cache.n_ctx)
		{
			fprintf(stderr, "gemma_eval_token: pos %d exceeds n_ctx %d\n",
			        pos, cache.n_ctx);
			return 1;
		}

		const int ple_dim = (h.ple_dim > 0) ? h.ple_dim : 0;
		const int ple_total = ple_dim * h.n_layer;

		float* xp = cache.sx.data();

		layer_scratch s{};
		s.normed = cache.snormed.data();
		s.q = cache.sq.data();
		s.k_cur = cache.sk_cur.data();
		s.v_cur = cache.sv_cur.data();
		s.attn_out = cache.sattn.data();
		s.gate = cache.sgate.data();
		s.up = cache.sup.data();
		s.scores = cache.sscores.data();
		s.ple_gate = cache.sple_gate.data();
		s.ple_contrib = cache.sple_contrib.data();
		s.ple_for_layer = nullptr;

		// Gemma scales the embedding by sqrt(n_embd); llama does not.
		const float embd_scale = (h.arch == "llama")
			? 1.0f
			: std::sqrt(static_cast<float>(h.n_embd));
		nnc_embed_row_bf16(xp, f.token_embd->data, token_id,
		                   static_cast<size_t>(h.n_embd), embd_scale);

		if (ple_dim > 0 && f.per_layer_token_embd && f.per_layer_model_proj && f.per_layer_proj_norm)
		{
			float* ple_inputs = cache.sple_inputs.data();
			float* bproj = cache.sple_proj.data();

			const float tok_scale = std::sqrt(static_cast<float>(ple_dim));
			const nnc_bf16_t* row = static_cast<const nnc_bf16_t*>(f.per_layer_token_embd->data)
				+ static_cast<size_t>(token_id) * static_cast<size_t>(ple_total);
			nnc_bf16_to_f32_row(row, ple_inputs, static_cast<size_t>(ple_total));
			for (int i = 0; i < ple_total; ++i) ple_inputs[i] *= tok_scale;

			gw_gemv(f.per_layer_model_proj, xp, bproj);
			const float inv_sqrt_n_embd = 1.0f / std::sqrt(static_cast<float>(h.n_embd));
			for (int i = 0; i < ple_total; ++i) bproj[i] *= inv_sqrt_n_embd;

			const auto gamma = static_cast<const float*>(f.per_layer_proj_norm->data);
			for (int li = 0; li < h.n_layer; ++li)
			{
				float* slc = bproj + static_cast<size_t>(li) * ple_dim;
				rmsnorm_with_gamma(slc, slc, ple_dim, gamma, h.rms_eps);
			}

			const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
			for (int i = 0; i < ple_total; ++i)
				ple_inputs[i] = (bproj[i] + ple_inputs[i]) * inv_sqrt2;
		}

		// layer_forward expects a std::vector<float>& for the residual
		// stream. Wrap our cached buffer in a non-owning view; we never
		// resize it inside the loop.
		// To avoid touching layer_forward's signature, use a
		// std::vector that aliases cache.sx in place of the per-call
		// allocation. The simplest correct option is to operate on
		// cache.sx directly via a small adapter that exposes data().
		// We reuse cache.sx as the actual storage by passing a
		// reference to it.
		std::vector<float>& xv = cache.sx;

		for (int li = 0; li < h.n_layer; ++li)
		{
			s.ple_for_layer = (ple_dim > 0 && f.per_layer_token_embd)
				                  ? cache.sple_inputs.data() + static_cast<size_t>(li) * ple_dim
				                  : nullptr;
			layer_forward(f, cache, li, pos, xv, s);
		}

		if (f.output_norm)
			rmsnorm_with_gamma(xp, xp, h.n_embd,
			                   static_cast<const float*>(f.output_norm->data), h.rms_eps);

		return 0;
	}
}

int gemma_eval_token(const gemma_file& f, gemma_kv_cache& cache,
                     const int token_id, const int pos, float* logits)
{
	const auto& h = f.hparams;
	const int rc = gemma_forward_to_x(f, cache, token_id, pos);
	if (rc != 0) return rc;

	const nnc_tensor* lm_head = f.output ? f.output : f.token_embd;
	gw_gemv(lm_head, cache.sx.data(), logits);
	if (h.final_logit_softcap > 0)
		nnc_softcap_f32(logits, logits, h.n_vocab, h.final_logit_softcap);

	cache.cur_pos = pos + 1;
	return 0;
}

// Same as gemma_eval_token but returns the argmax token id directly,
// without ever materialising the full vocab-sized logits buffer. Used
// by greedy decode (the chat REPL and gemma_generate). Soft-cap is
// monotonic so it has no effect on argmax and is skipped.
int gemma_eval_token_argmax(const gemma_file& f, gemma_kv_cache& cache,
                            const int token_id, const int pos, int* out_argmax)
{
	const auto& h = f.hparams;
	const int rc = gemma_forward_to_x(f, cache, token_id, pos);
	if (rc != 0) return rc;

	const nnc_tensor* lm_head = f.output ? f.output : f.token_embd;
	const int best = gw_argmax(lm_head, cache.sx.data());

	cache.cur_pos = pos + 1;
	if (out_argmax) *out_argmax = best;
	return 0;
}

// ----------------------------------------------------------------------------
// Greedy generation loop.
// ----------------------------------------------------------------------------

int gemma_generate(const gemma_file& f, const std::vector<int>& prompt_tokens,
                   const int n_predict, const int n_ctx)
{
	if (prompt_tokens.empty())
	{
		fprintf(stderr, "gemma_generate: empty prompt\n");
		return 1;
	}
	const auto& h = f.hparams;
	if (n_predict < 0 || n_ctx <= 0 ||
		prompt_tokens.size() > static_cast<size_t>(n_ctx - n_predict))
	{
		fprintf(stderr, "gemma_generate: prompt(%zu) + n_predict(%d) > n_ctx(%d)\n",
		        prompt_tokens.size(), n_predict, n_ctx);
		return 1;
	}

	gemma_kv_cache cache;
	if (!gemma_kv_init(f, cache, n_ctx)) return 1;

	// Print prompt.
	printf("\nprompt (%zu tokens):\n", prompt_tokens.size());
	for (size_t i = 0; i < prompt_tokens.size(); ++i)
	{
		const int t = prompt_tokens[i];
		const char* s = (t >= 0 && t < static_cast<int>(f.vocab_tokens.size()))
			                ? f.vocab_tokens[t].c_str()
			                : "?";
		printf("  [%3zu] id=%-7d '%s'\n", i, t, s);
	}

	// Prefill (greedy: use argmax-only path so we never materialise the
	// full vocab-sized logits buffer for any prefill or generate step).
	int next = -1;
	const int64_t t_pre0 = nnc_time_us();
	for (size_t i = 0; i < prompt_tokens.size(); ++i)
	{
		const int rc = gemma_eval_token_argmax(f, cache, prompt_tokens[i],
		                                       static_cast<int>(i), &next);
		if (rc != 0)
		{
			gemma_kv_free(cache);
			return rc;
		}
	}
	const double pre_ms = (nnc_time_us() - t_pre0) / 1000.0;
	printf("\nprefill: %zu tokens in %.1f ms (%.1f ms/tok)\n",
	       prompt_tokens.size(), pre_ms, pre_ms / prompt_tokens.size());

	printf("\ngenerated:\n");
	const int64_t t_gen0 = nnc_time_us();
	for (int g = 0; g < n_predict; ++g)
	{
		const int pos = static_cast<int>(prompt_tokens.size()) + g;
		const char* s = (next >= 0 && next < static_cast<int>(f.vocab_tokens.size()))
			                ? f.vocab_tokens[next].c_str()
			                : "?";
		printf("  [%3d] id=%-7d '%s'\n", g, next, s);
		fflush(stdout);
		if (next == f.eos_id) break;
		if (g + 1 >= n_predict) break;
		const int rc = gemma_eval_token_argmax(f, cache, next, pos, &next);
		if (rc != 0)
		{
			gemma_kv_free(cache);
			return rc;
		}
	}
	const double gen_ms = (nnc_time_us() - t_gen0) / 1000.0;
	printf("\ngenerate: %d tokens in %.1f ms (%.1f ms/tok)\n",
	       n_predict, gen_ms, gen_ms / std::max(1, n_predict));

	gemma_kv_free(cache);
	return 0;
}

// ----------------------------------------------------------------------------
// Backwards-compat single-token forward (now delegates to the cached path).
// ----------------------------------------------------------------------------

int gemma_forward_one(const gemma_file& f, const int token_id)
{
	const auto& h = f.hparams;
	if (token_id < 0 || token_id >= h.n_vocab)
	{
		fprintf(stderr, "gemma_forward_one: token_id %d out of range\n", token_id);
		return 1;
	}
	if (!f.token_embd)
	{
		fprintf(stderr, "no token_embd\n");
		return 1;
	}

	gemma_kv_cache cache;
	if (!gemma_kv_init(f, cache, 1)) return 1;
	std::vector<float> logits(h.n_vocab);

	printf("gemma_forward_one: token_id=%d ('%s')  n_layer=%d n_embd=%d n_vocab=%d\n",
	       token_id,
	       (token_id < static_cast<int>(f.vocab_tokens.size())) ? f.vocab_tokens[token_id].c_str() : "?",
	       h.n_layer, h.n_embd, h.n_vocab);

	const int64_t t0 = nnc_time_us();
	const int rc = gemma_eval_token(f, cache, token_id, 0, logits.data());
	const double ms = (nnc_time_us() - t0) / 1000.0;
	if (rc != 0)
	{
		gemma_kv_free(cache);
		return rc;
	}

	print_stats("logits (post-softcap)", logits.data(), h.n_vocab);

	struct entry
	{
		float v;
		int id;
	};
	std::vector<entry> top;
	top.reserve(h.n_vocab);
	for (int i = 0; i < h.n_vocab; ++i) top.push_back({logits[i], i});
	std::partial_sort(top.begin(), top.begin() + 5, top.end(),
	                  [](const entry& a, const entry& b) { return a.v > b.v; });
	printf("\ntop 5 logits:\n");
	for (int i = 0; i < 5; ++i)
	{
		const auto& e = top[i];
		const char* tok = (e.id < static_cast<int>(f.vocab_tokens.size()))
			                  ? f.vocab_tokens[e.id].c_str()
			                  : "?";
		printf("  [%2d]  id=%-7d logit=%+8.4f  token='%s'\n", i, e.id, e.v, tok);
	}
	printf("\ntotal forward: %.1f ms\n", ms);
	gemma_kv_free(cache);
	return 0;
}

// ----------------------------------------------------------------------------
// SentencePiece-style BPE tokenizer.
//
// Pre-tokenization: replace ' ' with U+2581 (▁, UTF-8 0xE2 0x96 0x81).
// Initial split: each UTF-8 codepoint is one piece. Then repeatedly find
// the adjacent pair with the lowest merge_rank ("lhs<space>rhs" -> rank
// position in the merges array) and merge. Stop when no more mergeable
// pairs exist. Map final pieces to vocab ids; unknown -> unk_id.
// ----------------------------------------------------------------------------

namespace
{
	// Length in bytes of a UTF-8 codepoint starting at the given lead byte.
	int utf8_len(const unsigned char c)
	{
		if (c < 0x80) return 1;
		if ((c & 0xE0) == 0xC0) return 2;
		if ((c & 0xF0) == 0xE0) return 3;
		if ((c & 0xF8) == 0xF0) return 4;
		return 1; // invalid lead -> consume one byte
	}

	void split_utf8_codepoints(const std::string& s, std::vector<std::string>& out)
	{
		out.clear();
		size_t i = 0;
		while (i < s.size())
		{
			const int n = utf8_len(static_cast<unsigned char>(s[i]));
			const size_t take = (i + n <= s.size()) ? n : (s.size() - i);
			out.emplace_back(s.data() + i, take);
			i += take;
		}
	}

	std::string sub_spaces(const std::string& in)
	{
		// Replace ' ' with the 3-byte ▁ (U+2581).
		std::string out;
		out.reserve(in.size() + in.size() / 4);
		for (const char c : in)
		{
			if (c == ' ') { out.append("\xE2\x96\x81", 3); }
			else out.push_back(c);
		}
		return out;
	}

	// GPT-2 byte-to-unicode mapping. Each byte 0..255 maps to a printable
	// Unicode codepoint, encoded as UTF-8. This is the same mapping as
	// the original GPT-2 BPE tokenizer (and Llama-3's, which derives from it).
	const std::array<std::string, 256>& byte_to_unicode_table()
	{
		static const auto table = []
		{
			std::array<std::string, 256> t;
			// Printable ranges that map to themselves: '!'..'~', '¡'..'¬', '®'..'ÿ'.
			std::vector<int> bs;
			for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
			for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
			for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
			std::vector<int> cs = bs;
			int n = 0;
			for (int b = 0; b < 256; ++b)
			{
				if (std::find(bs.begin(), bs.end(), b) == bs.end())
				{
					bs.push_back(b);
					cs.push_back(256 + n);
					++n;
				}
			}
			auto encode_utf8 = [](int cp) -> std::string
			{
				std::string s;
				if (cp < 0x80) { s.push_back((char)cp); }
				else if (cp < 0x800)
				{
					s.push_back((char)(0xC0 | (cp >> 6)));
					s.push_back((char)(0x80 | (cp & 0x3F)));
				}
				else
				{
					s.push_back((char)(0xE0 | (cp >> 12)));
					s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
					s.push_back((char)(0x80 | (cp & 0x3F)));
				}
				return s;
			};
			for (size_t i = 0; i < bs.size(); ++i)
				t[bs[i]] = encode_utf8(cs[i]);
			return t;
		}();
		return table;
	}

	std::string bytes_to_unicode(const std::string& in)
	{
		const auto& tab = byte_to_unicode_table();
		std::string out;
		out.reserve(in.size() * 2);
		for (unsigned char b : in) out.append(tab[b]);
		return out;
	}

	std::string restore_spaces(const std::string& in)
	{
		std::string out;
		out.reserve(in.size());
		for (size_t i = 0; i < in.size();)
		{
			if (i + 3 <= in.size() &&
				static_cast<unsigned char>(in[i]) == 0xE2 &&
				static_cast<unsigned char>(in[i + 1]) == 0x96 &&
				static_cast<unsigned char>(in[i + 2]) == 0x81)
			{
				out.push_back(' ');
				i += 3;
			}
			else
			{
				out.push_back(in[i]);
				++i;
			}
		}
		return out;
	}
}

std::vector<int> gemma_tokenize(const gemma_file& f, const std::string& text,
                                const bool add_bos)
{
	std::vector<int> out;
	if (add_bos && f.bos_id >= 0) out.push_back(f.bos_id);

	if (text.empty()) return out;

	// GPT-2 / Llama BPE path (tokenizer.ggml.model="gpt2"): byte-to-unicode
	// encode input, then merge by rank from merges[]. We skip the GPT-4
	// pretokenize regex; for chat-style English input the merges naturally
	// resolve word boundaries via 'Ġ' (= encoded space) prefixes.
	if (f.tokenizer_model == "gpt2" && !f.merges.empty())
	{
		const std::string enc = bytes_to_unicode(text);
		std::vector<std::string> pieces;
		split_utf8_codepoints(enc, pieces);

		auto pair_key = [](const std::string& a, const std::string& b)
		{
			std::string k;
			k.reserve(a.size() + 1 + b.size());
			k.append(a);
			k.push_back(' ');
			k.append(b);
			return k;
		};

		while (pieces.size() >= 2)
		{
			int best_rank = std::numeric_limits<int>::max();
			size_t best_i = pieces.size();
			for (size_t i = 0; i + 1 < pieces.size(); ++i)
			{
				auto it = f.merge_rank.find(pair_key(pieces[i], pieces[i + 1]));
				if (it != f.merge_rank.end() && it->second < best_rank)
				{
					best_rank = it->second;
					best_i = i;
				}
			}
			if (best_i == pieces.size()) break;
			pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
			pieces.erase(pieces.begin() + best_i + 1);
		}

		out.reserve(out.size() + pieces.size());
		for (const auto& p : pieces)
		{
			auto it = f.token_to_id.find(p);
			if (it != f.token_to_id.end())
				out.push_back(it->second);
			else if (f.unk_id >= 0)
				out.push_back(f.unk_id);
		}
		return out;
	}

	// Optional leading-space rule (Gemma 3n: add_space_prefix=false, so off).
	std::string s = f.add_space_prefix ? (std::string(" ") + text) : text;
	s = sub_spaces(s);

	// SentencePiece-BPE path (tokenizer.ggml.model="llama"): priority-merge
	// adjacent symbols by vocab score (higher = merge first). Used by gemma3
	// and llama models. Falls back to <0xHH> byte tokens for unknown bytes.
	if (f.merges.empty() && !f.vocab_scores.empty())
	{
		// Pre-split into UTF-8 codepoint symbols, kept as a doubly-linked
		// list via prev/next index arrays so we can erase merged pairs in
		// O(1).
		std::vector<std::string> sym;
		split_utf8_codepoints(s, sym);
		const int N = (int)sym.size();
		std::vector<int> prev(N), next(N);
		for (int i = 0; i < N; ++i) { prev[i] = i - 1; next[i] = i + 1; }
		next[N - 1] = -1;

		struct Bigram
		{
			float score;
			int left;   // index of left symbol
			int rev;    // revision counter to invalidate stale entries
			size_t len; // total bytes
			bool operator<(const Bigram& o) const
			{
				if (score != o.score) return score < o.score; // max-heap
				return left > o.left; // tie-break: leftmost first
			}
		};
		std::priority_queue<Bigram> pq;
		std::vector<int> rev(N, 0);

		auto try_push = [&](int left)
		{
			if (left < 0) return;
			const int right = next[left];
			if (right < 0) return;
			std::string merged = sym[left] + sym[right];
			auto it = f.token_to_id.find(merged);
			if (it == f.token_to_id.end()) return;
			const int id = it->second;
			const float sc = (id >= 0 && id < (int)f.vocab_scores.size())
				                 ? f.vocab_scores[id]
				                 : 0.0f;
			pq.push({sc, left, rev[left], merged.size()});
		};

		for (int i = 0; i + 1 < N; ++i) try_push(i);

		while (!pq.empty())
		{
			Bigram b = pq.top();
			pq.pop();
			if (b.rev != rev[b.left]) continue;          // stale
			const int right = next[b.left];
			if (right < 0) continue;                     // stale
			if (sym[b.left].size() + sym[right].size() != b.len) continue; // stale

			sym[b.left] = sym[b.left] + sym[right];
			rev[b.left]++;
			// unlink right
			const int rr = next[right];
			next[b.left] = rr;
			if (rr >= 0) prev[rr] = b.left;
			// invalidate right
			rev[right] = -1;
			// push new bigrams on either side of merged left
			try_push(b.left);
			try_push(prev[b.left]);
		}

		// Walk the final linked list, emitting ids (with byte fallback).
		for (int i = 0; i >= 0 && i < N; i = next[i])
		{
			if (rev[i] < 0) continue;
			auto it = f.token_to_id.find(sym[i]);
			if (it != f.token_to_id.end())
			{
				out.push_back(it->second);
				continue;
			}
			// Byte fallback per UTF-8 byte.
			for (unsigned char c : sym[i])
			{
				char hex[8];
				std::snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned)c);
				auto bt = f.token_to_id.find(hex);
				if (bt != f.token_to_id.end()) out.push_back(bt->second);
				else if (f.unk_id >= 0) out.push_back(f.unk_id);
			}
		}
		return out;
	}

	// Split into codepoint pieces.
	std::vector<std::string> pieces;
	split_utf8_codepoints(s, pieces);

	// Greedy lowest-rank merge.
	auto pair_key = [](const std::string& a, const std::string& b)
	{
		std::string k;
		k.reserve(a.size() + 1 + b.size());
		k.append(a);
		k.push_back(' ');
		k.append(b);
		return k;
	};

	while (pieces.size() >= 2)
	{
		int best_rank = std::numeric_limits<int>::max();
		size_t best_i = pieces.size();
		for (size_t i = 0; i + 1 < pieces.size(); ++i)
		{
			auto it = f.merge_rank.find(pair_key(pieces[i], pieces[i + 1]));
			if (it != f.merge_rank.end() && it->second < best_rank)
			{
				best_rank = it->second;
				best_i = i;
			}
		}
		if (best_i == pieces.size()) break;
		pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
		pieces.erase(pieces.begin() + best_i + 1);
	}

	// Map to ids.
	out.reserve(out.size() + pieces.size());
	for (const auto& p : pieces)
	{
		auto it = f.token_to_id.find(p);
		if (it != f.token_to_id.end())
		{
			out.push_back(it->second);
		}
		else if (f.unk_id >= 0)
		{
			out.push_back(f.unk_id);
		}
		else
		{
			// No unk_id configured — warn (once per call) and substitute id 0
			// rather than silently turning unknown pieces into a real token.
			thread_local bool warned = false;
			if (!warned)
			{
				fprintf(stderr,
				        "warning: no unk_id in vocab; unknown piece '%s' -> 0\n",
				        p.c_str());
				warned = true;
			}
			out.push_back(0);
		}
	}
	return out;
}

std::string gemma_detokenize(const gemma_file& f, const std::vector<int>& tokens)
{
	std::string acc;
	acc.reserve(tokens.size() * 4);
	for (const int t : tokens)
	{
		if (t < 0 || t >= static_cast<int>(f.vocab_tokens.size())) continue;
		acc.append(f.vocab_tokens[t]);
	}
	if (f.tokenizer_model == "gpt2")
	{
		// Reverse byte-to-unicode mapping: walk the encoded string,
		// pull off one Unicode codepoint at a time, and emit the byte
		// it represents.
		const auto& tab = byte_to_unicode_table();
		std::unordered_map<std::string, unsigned char> rev;
		for (int b = 0; b < 256; ++b) rev.emplace(tab[b], (unsigned char)b);
		std::string out;
		out.reserve(acc.size());
		size_t i = 0;
		while (i < acc.size())
		{
			const int n = utf8_len((unsigned char)acc[i]);
			const size_t take = (i + n <= acc.size()) ? n : (acc.size() - i);
			std::string cp(acc.data() + i, take);
			auto it = rev.find(cp);
			if (it != rev.end()) out.push_back((char)it->second);
			else out.append(cp); // fallback: pass through (e.g. special tokens)
			i += take;
		}
		return out;
	}
	return restore_spaces(acc);
}

// ----------------------------------------------------------------------------
// Q8_0 in-place quantisation. Walks the weight tensors that dominate the
// decode/prefill GEMV time, replaces their `data` pointer with a fresh Q8_0
// allocation (qs[] then scales[]), and flips `tensor->type`. Subsequent
// calls to gw_gemv / gw_argmax then route through nnc_gemv_q8_0_f32x.
// ----------------------------------------------------------------------------

namespace
{
	// Quantise a single BF16 [rows, cols] weight tensor in place. Returns
	// false on allocation failure or unsupported shape (cols not a positive
	// multiple of 32). `label` is for the optional log line.
	bool quantize_one(gemma_file& f, nnc_tensor* T, const char* label,
	                  size_t& bf16_bytes, size_t& q8_bytes)
	{
		if (!T) return true; // missing optional tensor
		if (T->type != NNC_TYPE_BF16) return true; // already converted / unsupported
		if (T->n_dims != 2) return true;
		const int64_t cols = T->ne[0];
		const int64_t rows = T->ne[1];
		if (cols <= 0 || rows <= 0 || (cols % 32) != 0) return true;

		const size_t qs_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols);
		const size_t scales_count = static_cast<size_t>(rows) * static_cast<size_t>(cols / 32);
		const size_t total = qs_bytes + scales_count * sizeof(float);

		auto buf = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t[total]);
		if (!buf)
		{
			fprintf(stderr, "gemma_quantize_q8_0: alloc %zu bytes failed for %s\n",
			        total, label ? label : "<tensor>");
			return false;
		}
		auto* qs = reinterpret_cast<int8_t*>(buf.get());
		auto* scales = reinterpret_cast<float*>(buf.get() + qs_bytes);

		nnc_quantize_bf16_to_q8_0(static_cast<const nnc_bf16_t*>(T->data),
		                          qs, scales,
		                          static_cast<uint32_t>(rows),
		                          static_cast<uint32_t>(cols));

		bf16_bytes += qs_bytes * 2; // BF16 was 2 bytes/elem
		q8_bytes += total;

		T->data = buf.get();
		T->type = NNC_TYPE_Q8_0;
		f.q8_buffers.push_back(std::move(buf));
		return true;
	}
}

bool gemma_quantize_q8_0(gemma_file& f)
{
	size_t bf16_bytes = 0, q8_bytes = 0;
	int n_quant = 0;

	auto take = [&](nnc_tensor* T, const char* label) -> bool
	{
		const bool was_bf16 = (T && T->type == NNC_TYPE_BF16);
		if (!quantize_one(f, T, label, bf16_bytes, q8_bytes)) return false;
		if (was_bf16 && T->type == NNC_TYPE_Q8_0) ++n_quant;
		return true;
	};

	for (const auto& L : f.layers)
	{
		if (!take(L.attn_q, "attn_q")) return false;
		if (!take(L.attn_k, "attn_k")) return false;
		if (!take(L.attn_v, "attn_v")) return false;
		if (!take(L.attn_output, "attn_output")) return false;
		if (!take(L.ffn_gate, "ffn_gate")) return false;
		if (!take(L.ffn_up, "ffn_up")) return false;
		if (!take(L.ffn_down, "ffn_down")) return false;
		// PLE per-layer projections: small but keep them BF16-or-Q8 uniform.
		if (!take(L.inp_gate, "inp_gate")) return false;
		if (!take(L.proj, "proj")) return false;
	}
	// NOTE: f.token_embd is intentionally NOT quantised. It doubles as the
	// input embedding table (read via nnc_embed_row_bf16) AND the lm_head
	// projection. The embedding-lookup path expects raw BF16 rows; flipping
	// the type to Q8_0 would feed garbage into x. Cost: lm_head stays BF16
	// (~17% of decode), so the realised speedup is below the theoretical
	// 1.78x BW reduction.
	if (!take(f.per_layer_model_proj, "per_layer_model_proj")) return false;
	// llama-style separate lm_head: safe to quantize (not used as embedding lookup).
	if (!take(f.output, "output")) return false;

	const double bf16_mb = static_cast<double>(bf16_bytes) / (1024.0 * 1024.0);
	const double q8_mb = static_cast<double>(q8_bytes) / (1024.0 * 1024.0);
	fprintf(stderr, "gemma_quantize_q8_0: %d tensors, %.1f MB BF16 -> %.1f MB Q8_0 (%.2fx)\n",
	        n_quant, bf16_mb, q8_mb,
	        bf16_bytes ? (bf16_mb / q8_mb) : 0.0);
	return true;
}
