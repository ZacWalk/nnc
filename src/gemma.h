// nnc — Gemma 3n (E2B / E4B) model loader.
//
// Reads a `general.architecture == "gemma4"` GGUF file via gguf_mmap,
// pulls the hyperparameters out of the metadata KV table, and creates
// nnc_tensor handles whose `data` pointers reference the mmapped weight
// bytes directly (no copy).
//
// Inference is not yet implemented; this header gives downstream code
// (and the --gemma-info CLI) something to load and inspect.

#pragma once

#include "gguf.h"
#include "runtime.h"

#include <string>
#include <unordered_map>
#include <vector>

struct gemma_layer
{
	// Pre-norms (RMSNorm gamma, F32 vectors).
	nnc_tensor* attn_norm = nullptr; // [n_embd]
	nnc_tensor* ffn_norm = nullptr; // [n_embd]
	nnc_tensor* post_attention_norm = nullptr; // [n_embd]
	nnc_tensor* post_ffw_norm = nullptr; // [n_embd]
	nnc_tensor* post_norm = nullptr; // [n_embd]
	nnc_tensor* attn_q_norm = nullptr; // [head_dim]
	nnc_tensor* attn_k_norm = nullptr; // [head_dim]

	// Attention projections (BF16 matrices).
	nnc_tensor* attn_q = nullptr; // [n_embd, n_head*head_dim]
	nnc_tensor* attn_k = nullptr; // [n_embd, n_head_kv*head_dim]
	nnc_tensor* attn_v = nullptr; // [n_embd, n_head_kv*head_dim]
	nnc_tensor* attn_output = nullptr; // [n_head*head_dim, n_embd]

	// MLP projections (BF16).
	nnc_tensor* ffn_gate = nullptr; // [n_embd, n_ff]
	nnc_tensor* ffn_up = nullptr; // [n_embd, n_ff]
	nnc_tensor* ffn_down = nullptr; // [n_ff, n_embd]

	// Per-layer-input embedding gate (Gemma 3n PLE).
	nnc_tensor* inp_gate = nullptr; // [n_embd, ple_dim]
	nnc_tensor* proj = nullptr; // [ple_dim, n_embd]
	nnc_tensor* layer_output_scale = nullptr; // [1]

	// Per-layer FFN size (Gemma E2B has a per-layer array; for E4B it is
	// constant). Stored here so attention/MLP graph construction does not
	// have to hit the hparams table.
	int n_ff = 0;
	bool sliding_window = false;
};

struct gemma_hparams
{
	std::string arch; // "gemma4"
	int n_vocab = 0; // tokenizer.ggml.tokens.size()
	int n_ctx = 0; // gemma4.context_length
	int n_embd = 0; // gemma4.embedding_length
	int n_layer = 0; // gemma4.block_count
	int n_head = 0; // gemma4.attention.head_count
	int n_head_kv = 0; // gemma4.attention.head_count_kv (GQA)
	int head_dim = 0; // gemma4.attention.key_length
	int head_dim_swa = 0; // gemma4.attention.key_length_swa
	int rope_dim = 0; // gemma4.rope.dimension_count
	int rope_dim_swa = 0; // gemma4.rope.dimension_count_swa
	float rope_freq_base = 0; // gemma4.rope.freq_base
	float rope_freq_base_swa = 0; // gemma4.rope.freq_base_swa
	float rms_eps = 0; // gemma4.attention.layer_norm_rms_epsilon
	int sliding_window = 0; // gemma4.attention.sliding_window
	int shared_kv_layers = 0; // gemma4.attention.shared_kv_layers
	int ple_dim = 0; // gemma4.embedding_length_per_layer_input
	float final_logit_softcap = 0; // gemma4.final_logit_softcapping
	int file_type = 0; // raw GGUF file_type (32 = mostly BF16)
};

struct gemma_model
{
	// (Reserved for future runtime-side state — KV cache, scratch
	// buffers, etc. — once the per-token graph is wired up.)
	int reserved = 0;
};

struct gemma_file
{
	gguf_file gguf; // owns the mmap
	gemma_hparams hparams;

	// Vocabulary as parsed from tokenizer.ggml.tokens (UTF-8 strings).
	std::vector<std::string> vocab_tokens;
	std::vector<float> vocab_scores;
	std::vector<int32_t> vocab_token_types;
	int bos_id = -1, eos_id = -1, unk_id = -1, pad_id = -1;

	// BPE tokenizer tables (Gemma uses SentencePiece-style BPE with the
	// space-as-▁ (U+2581) convention). `merges` are "lhs<space>rhs"
	// strings; their position in the array is the merge priority (lower
	// rank = applied earlier). We keep precomputed lookup tables so the
	// per-token tokenize/detokenize calls are O(merge-passes * tokens).
	std::vector<std::string> merges;
	std::unordered_map<std::string, int> token_to_id; // piece -> vocab id
	std::unordered_map<std::string, int> merge_rank; // "lhs rhs" -> rank
	bool add_bos_token = true;
	bool add_space_prefix = false;

	// Top-level weights (BF16 unless noted).
	nnc_tensor* token_embd = nullptr; // [n_embd, n_vocab]
	nnc_tensor* per_layer_token_embd = nullptr; // [ple_dim*n_layer, n_vocab]
	nnc_tensor* per_layer_model_proj = nullptr; // [n_embd, ple_dim*n_layer]
	nnc_tensor* per_layer_proj_norm = nullptr; // [ple_dim] (F32)
	nnc_tensor* output_norm = nullptr; // [n_embd]  (F32; if absent, tied to token_embd norm)
	nnc_tensor* rope_freqs = nullptr; // [rope_dim/2]  precomputed (F32)

	std::vector<gemma_layer> layers;

	// nnc context that owns the tensor descriptors (NOT the weight bytes,
	// which live in the gguf mmap). Sized to fit ~601 tensor headers.
	nnc_context* ctx = nullptr;
};

bool gemma_load(const std::string& path, gemma_file& out);
void gemma_free(gemma_file& f);

// Pretty-print parsed hparams, vocab summary, and which expected
// tensors were located vs missing.
void gemma_print_info(const gemma_file& f);

// Run a partial forward pass on a single token (default: BOS) through
// layer 0 of the model: embed -> attn_norm * gamma -> Q projection ->
// attn_q_norm * gamma. Prints min/max/mean/rms of each intermediate
// activation. Used as a smoke test that the BF16 weights and kernels
// agree on real model data.
int gemma_probe(const gemma_file& f, int token_id);

// Run the full single-token forward pass through every layer
// (attention is degenerate at pos=0: softmax over one key = 1, so the
// attention output is just V projected by attn_output). PLE / shared-KV
// / sliding-window are stubbed for this first pass — the goal is to
// verify the loader + kernels can drive the entire stack and produce
// finite logits. Prints layer-0 / mid / final activation stats and the
// top-5 logits.
int gemma_forward_one(const gemma_file& f, int token_id);

// Per-layer key/value cache. K/V are stored as F32 in token-major order:
// k[li][pos * kv_dim_li + i]. kv_dim_li can differ per layer (Gemma 3n
// E2B uses head_dim=512 for full layers and 256 for sliding-window
// layers, with n_kv_heads=1).
struct gemma_kv_cache
{
	int n_ctx = 0; // max positions
	int cur_pos = 0; // next slot to fill
	std::vector<int> kv_dim_per_layer;
	std::vector<std::vector<float>> k; // [n_layer][n_ctx * kv_dim_li]
	std::vector<std::vector<float>> v;
};

bool gemma_kv_init(const gemma_file& f, gemma_kv_cache& cache, int n_ctx);
void gemma_kv_free(gemma_kv_cache& cache);

// Evaluate one token at `pos` (must equal cache.cur_pos). Appends K/V
// to the cache, computes attention against all cached positions in the
// per-layer window (full-context for global layers, last
// `hparams.sliding_window` positions for SWA layers), and writes
// post-softcap logits into `logits` (size = n_vocab). Advances
// cache.cur_pos.
int gemma_eval_token(const gemma_file& f, gemma_kv_cache& cache,
                     int token_id, int pos, float* logits);

// Same as gemma_eval_token but returns argmax(logits) directly via
// `out_argmax`, never materialising the full vocab-sized logits buffer
// or applying soft-cap (monotonic, so argmax-preserving). Used by
// greedy decode in the chat REPL and gemma_generate. Saves ~1 MB of
// logits writes + a softcap pass per token at vocab=262144.
int gemma_eval_token_argmax(const gemma_file& f, gemma_kv_cache& cache,
                            int token_id, int pos, int* out_argmax);

// Greedy multi-token generation. Feeds `prompt_tokens` through the
// model (filling KV cache), then samples `n_predict` more tokens by
// argmax over the logits. Prints each generated token id + string and
// per-token timing. Returns 0 on success.
int gemma_generate(const gemma_file& f, const std::vector<int>& prompt_tokens,
                   int n_predict, int n_ctx);

// SentencePiece-style BPE tokenization of UTF-8 text using the model's
// merges. If `add_bos` is true (and bos_id >= 0), the BOS token is
// prepended. Spaces are replaced by U+2581 (▁) before merging.
std::vector<int> gemma_tokenize(const gemma_file& f, const std::string& text,
                                bool add_bos);

// Inverse of gemma_tokenize: concatenates the piece strings, then
// converts U+2581 back to ASCII space. Special tokens (BOS/EOS/...)
// are emitted as their literal piece string (e.g. "<bos>").
std::string gemma_detokenize(const gemma_file& f, const std::vector<int>& tokens);
