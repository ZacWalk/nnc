// nnc — CLI entry point and Gemma inference driver.
// Parses arguments, dispatches to --test and the various --gguf-*/--gemma-*
// inspection modes, otherwise loads a Gemma GGUF model and runs an
// interactive chat REPL against the nnc runtime.

#include "runtime.h"

#include "gemma.h"
#include "gguf.h"
#include "sys.h"
#include "utils.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int run_tests(); // tests.cpp

// Build a deduplicated list of .gguf files found under the standard
// search roots: ./models, $HOME/.lmstudio/models, $HOME/models. Returns
// at most 26 entries (one per a-z letter) in the order discovered.
static std::vector<std::string> collect_model_candidates()
{
	std::vector<std::string> out;
	auto add_from = [&](const std::string& root)
	{
		if (root.empty()) return;
		std::vector<std::string> found;
		sys_list_files_recursive(root.c_str(), ".gguf", found);
		for (auto& p : found)
		{
			bool dup = false;
			for (const auto& e : out) if (e == p) { dup = true; break; }
			if (!dup) out.push_back(std::move(p));
			if (out.size() >= 26) return;
		}
	};

	add_from("models");
	const std::string home = sys_home_dir();
	if (!home.empty())
	{
#if defined(_WIN32)
		add_from(home + "\\.lmstudio\\models");
		add_from(home + "\\models");
#else
		add_from(home + "/.lmstudio/models");
		add_from(home + "/models");
#endif
	}
	return out;
}

// Print the candidate list and read a single letter (a, b, c, ...) from
// stdin. Returns the picked path, or empty string on EOF / invalid input.
static std::string prompt_for_model(const std::vector<std::string>& cands)
{
	if (cands.empty()) return {};
	printf("nnc: select a model to load:\n");
	for (size_t i = 0; i < cands.size(); ++i)
	{
		printf("  %c) %s\n", static_cast<char>('a' + i), cands[i].c_str());
	}
	printf("\n> ");
	fflush(stdout);

	std::string line;
	if (!std::getline(std::cin, line)) return {};
	// Trim leading whitespace.
	size_t p = 0;
	while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
	if (p >= line.size()) return {};
	const char c = line[p];
	const char lo = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
	if (lo < 'a' || lo > 'z') return {};
	const size_t idx = static_cast<size_t>(lo - 'a');
	if (idx >= cands.size()) return {};
	return cands[idx];
}

// Collapse runs of digits in `name` to '*' so that
// "blk.0.attn_q.weight" and "blk.31.attn_q.weight" map to the same
// pattern key, then each pattern can be printed with a count.
static std::string tensor_name_pattern(const std::string& name)
{
	std::string out;
	out.reserve(name.size());
	bool prev_digit = false;
	for (char c : name)
	{
		if (c >= '0' && c <= '9')
		{
			if (!prev_digit) out.push_back('*');
			prev_digit = true;
		}
		else
		{
			out.push_back(c);
			prev_digit = false;
		}
	}
	return out;
}

// Print a model digest tailored to "what would nnc need to support to
// load this?". Reads only the GGUF header + KV + tensor descriptors;
// no tensor data, no gemma-specific load path.
static int inspect_model_for_nnc(const char* path)
{
	gguf_file f{};
	if (!gguf_load(path, f)) return 1;

	printf("file:        %s\n", path);
	printf("gguf:        v%u, %llu KVs, %llu tensors, alignment=%llu\n",
	       f.version,
	       static_cast<unsigned long long>(f.kv_count),
	       static_cast<unsigned long long>(f.tensor_count),
	       static_cast<unsigned long long>(f.alignment));

	// --- top-level metadata ---------------------------------------------
	auto kv_str_or = [&](const char* key, const char* dflt) -> std::string
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v || v->type != GGUF_TYPE_STRING) return dflt;
		return v->str;
	};
	auto kv_u64_or = [&](const char* key, uint64_t dflt) -> uint64_t
	{
		const gguf_value* v = gguf_find_kv(f, key);
		if (!v) return dflt;
		switch (v->type)
		{
			case GGUF_TYPE_UINT8:
			case GGUF_TYPE_UINT16:
			case GGUF_TYPE_UINT32:
			case GGUF_TYPE_UINT64:
			case GGUF_TYPE_BOOL:
				return v->u64;
			case GGUF_TYPE_INT8:
			case GGUF_TYPE_INT16:
			case GGUF_TYPE_INT32:
			case GGUF_TYPE_INT64:
				return static_cast<uint64_t>(v->i64);
			default: return dflt;
		}
	};

	const std::string arch = kv_str_or("general.architecture", "(missing)");
	printf("architecture: %s\n", arch.c_str());
	printf("name:        %s\n", kv_str_or("general.name", "(none)").c_str());
	printf("file_type:   %llu\n",
	       static_cast<unsigned long long>(kv_u64_or("general.file_type", ~0ull)));

	// Vocab size if present (key naming varies by arch).
	const gguf_value* vocab_kv = gguf_find_kv(f, "tokenizer.ggml.tokens");
	if (vocab_kv && vocab_kv->type == GGUF_TYPE_ARRAY)
	{
		printf("vocab:       %zu tokens\n", vocab_kv->arr.size());
	}

	// Common per-arch hparam keys (we just probe by `<arch>.<suffix>`).
	if (arch != "(missing)")
	{
		auto pr_int = [&](const char* suffix)
		{
			const std::string key = arch + "." + suffix;
			const gguf_value* v = gguf_find_kv(f, key.c_str());
			if (!v) return;
			printf("  %-40s = %llu\n", key.c_str(),
			       static_cast<unsigned long long>(kv_u64_or(key.c_str(), 0)));
		};
		auto pr_f = [&](const char* suffix)
		{
			const std::string key = arch + "." + suffix;
			const gguf_value* v = gguf_find_kv(f, key.c_str());
			if (!v) return;
			double d = (v->type == GGUF_TYPE_FLOAT32 || v->type == GGUF_TYPE_FLOAT64)
				           ? v->f64
				           : 0.0;
			printf("  %-40s = %.6g\n", key.c_str(), d);
		};
		pr_int("context_length");
		pr_int("embedding_length");
		pr_int("block_count");
		pr_int("attention.head_count");
		pr_int("attention.head_count_kv");
		pr_int("attention.key_length");
		pr_int("attention.value_length");
		pr_int("feed_forward_length");
		pr_int("rope.dimension_count");
		pr_f("rope.freq_base");
		pr_f("attention.layer_norm_rms_epsilon");
	}

	// --- tensor type histogram + total bytes ----------------------------
	std::unordered_map<uint32_t, uint64_t> type_count;
	std::unordered_map<uint32_t, uint64_t> type_bytes;
	uint64_t total_bytes = 0;
	for (const auto& t : f.tensors)
	{
		type_count[t.ggml_type] += 1;
		const uint64_t nb = gguf_tensor_nbytes(t);
		type_bytes[t.ggml_type] += nb;
		total_bytes += nb;
	}
	printf("\ntensor types (%zu unique, %.2f MB total):\n",
	       type_count.size(), total_bytes / (1024.0 * 1024.0));
	for (const auto& kv : type_count)
	{
		printf("  %-10s %6llu tensors  %8.2f MB\n",
		       gguf_ggml_type_name(kv.first),
		       static_cast<unsigned long long>(kv.second),
		       type_bytes[kv.first] / (1024.0 * 1024.0));
	}

	// --- tensor name patterns (collapse blk.N.* -> blk.*.* etc.) --------
	std::unordered_map<std::string, uint64_t> pattern_count;
	std::vector<std::string> pattern_order;
	for (const auto& t : f.tensors)
	{
		const std::string p = tensor_name_pattern(t.name);
		if (pattern_count.emplace(p, 0).second) pattern_order.push_back(p);
		pattern_count[p] += 1;
	}
	printf("\ntensor name patterns (%zu unique):\n", pattern_order.size());
	for (const auto& p : pattern_order)
	{
		printf("  %4llu  %s\n",
		       static_cast<unsigned long long>(pattern_count[p]), p.c_str());
	}

	// --- nnc compatibility verdict --------------------------------------
	// Currently nnc supports gemma4 only, BF16/F16/F32/Q8_0 weight types.
	// (We can fold in Q4_K etc. once the kernels exist.)
	const bool arch_ok = (arch == "gemma4");
	bool types_ok = true;
	for (const auto& kv : type_count)
	{
		// Mirrors the dtype enum in runtime.h: F32=0, F16=1, BF16=30.
		// Q8_0 (=8) is the only quantised type we currently load (and
		// only via the in-process gemma_quantize_q8_0 conversion, not
		// from disk yet). Anything else would need new dequant code.
		const uint32_t t = kv.first;
		const bool ok = (t == 0 /*F32*/) || (t == 1 /*F16*/) || (t == 30 /*BF16*/);
		if (!ok)
		{
			types_ok = false;
			break;
		}
	}
	printf("\nnnc compatibility:\n");
	printf("  architecture supported: %s%s\n",
	       arch_ok ? "yes" : "no",
	       arch_ok ? "" : "  (only 'gemma4' is wired up today)");
	printf("  tensor dtypes supported: %s\n",
	       types_ok ? "yes (all F32/F16/BF16)" : "no  (see histogram above)");
	if (!arch_ok)
	{
		printf("  -> to load this, add a loader/forward for arch '%s' "
		       "alongside gemma.cpp.\n", arch.c_str());
	}
	if (!types_ok)
	{
		printf("  -> to load this, add a dequant path for the new tensor "
		       "types (e.g. Q4_K/Q5_K/Q6_K) in nn_ops.cpp / jit_ops.cpp.\n");
	}

	return 0;
}

// Parse a non-negative decimal int from `s`. Returns true on success and
// writes the value to *out. Rejects empty input, garbage trailing chars,
// negatives, and values >= INT_MAX.
static bool parse_int_strict(const char* s, int* out)
{
	if (!s || !*s) return false;
	errno = 0;
	char* end = nullptr;
	const long v = strtol(s, &end, 10);
	if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT_MAX) return false;
	*out = static_cast<int>(v);
	return true;
}

int main(int argc, char** argv)
{
	sys_init_crash_handlers();
	// Force UTF-8 on the Windows console so tokens containing U+2581 (▁),
	// em-dashes, etc. render correctly instead of as cp437 mojibake.
	// No-op on Linux.
	sys_init_console();

	// Pre-scan for global flags that aren't tied to a specific subcommand.
	// Q8_0 quantisation of the dominant weight tensors is ON by default;
	// pass `--bf16` (or `-bf16`) to keep weights as raw BF16. Applies to
	// any code path that calls gemma_load.
	bool quantize_q8 = true;
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--bf16") == 0 || strcmp(a, "-bf16") == 0)
		{
			quantize_q8 = false;
		}
		else if (strcmp(a, "--q8") == 0 || strcmp(a, "-q8") == 0)
		{
			quantize_q8 = true; // accepted for back-compat; this is the default
		}
	}

	auto maybe_quantize = [&](gemma_file& gf) -> bool
	{
		if (!quantize_q8) return true;
		return gemma_quantize_q8_0(gf);
	};

	// Pre-scan for an explicit `-m` / `--model` so we know whether to
	// run the interactive model picker on REPL startup.
	bool model_explicit = false;
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "-m") == 0 || strcmp(a, "--model") == 0)
		{
			model_explicit = true;
			break;
		}
	}

	// nnc --test / -test / /test : run all tests and exit.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--test") == 0 || strcmp(a, "-test") == 0 || strcmp(a, "/test") == 0)
		{
			return run_tests();
		}
	}

	// nnc --gguf-info <file> : parse a GGUF file's header + metadata +
	// tensor table and print them. No inference, no tensor data read.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gguf-info") == 0 || strcmp(a, "-gguf-info") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --gguf-info requires a file path\n");
				return 1;
			}
			gguf_file f{};
			if (!gguf_load(argv[i + 1], f))
				return 1;
			gguf_print_info(f);
			return 0;
		}
	}

	// nnc --gguf-stats <file> [name-substring] : mmap a GGUF file and
	// print min/max/mean and the first few values for each tensor whose
	// name contains the substring (default: every tensor). Verifies the
	// loader can decode F32/F16/BF16 weights end-to-end.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gguf-stats") == 0 || strcmp(a, "-gguf-stats") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --gguf-stats requires a file path\n");
				return 1;
			}
			const char* path = argv[i + 1];
			const char* needle = (i + 2 < argc) ? argv[i + 2] : "";
			extern int gguf_stats_main(const char* path, const char* needle);
			return gguf_stats_main(path, needle);
		}
	}

	// nnc --list-models : enumerate .gguf files under ./models,
	// $HOME/.lmstudio/models, and $HOME/models (recursive). Same source
	// list that the interactive picker uses on bare `nnc`.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--list-models") == 0 || strcmp(a, "-list-models") == 0)
		{
			const std::vector<std::string> cands = collect_model_candidates();
			if (cands.empty())
			{
				fprintf(stderr,
				        "nnc: no .gguf models found under ./models, "
				        "$HOME/.lmstudio/models, or $HOME/models.\n");
				return 1;
			}
			for (size_t j = 0; j < cands.size(); ++j)
			{
				printf("  %c) %s\n", static_cast<char>('a' + j), cands[j].c_str());
			}
			return 0;
		}
	}

	// nnc --inspect-model <file> : read the GGUF header + KV table +
	// tensor descriptors and print a digest aimed at "would nnc load
	// this, and if not, what would need to be added?". Cheap; does not
	// touch tensor data.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--inspect-model") == 0 || strcmp(a, "-inspect-model") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --inspect-model requires a file path\n");
				return 1;
			}
			return inspect_model_for_nnc(argv[i + 1]);
		}
	}

	// nnc --inspect-all : run --inspect-model over every model that
	// --list-models would surface. Useful for surveying which extra
	// architectures or tensor types are sitting on disk.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--inspect-all") == 0 || strcmp(a, "-inspect-all") == 0)
		{
			const std::vector<std::string> cands = collect_model_candidates();
			if (cands.empty())
			{
				fprintf(stderr, "nnc: no .gguf models found.\n");
				return 1;
			}
			for (size_t j = 0; j < cands.size(); ++j)
			{
				printf("\n========== %c) %s ==========\n",
				       static_cast<char>('a' + j), cands[j].c_str());
				inspect_model_for_nnc(cands[j].c_str());
			}
			return 0;
		}
	}

	// nnc --gemma-info <file> : load a Gemma 3n GGUF, parse its hparams
	// and tokenizer, build nnc_tensor descriptors for every weight, and
	// report which tensors were found. No inference yet.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-info") == 0 || strcmp(a, "-gemma-info") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --gemma-info requires a file path\n");
				return 1;
			}
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;
			gemma_print_info(gf);
			gemma_free(gf);
			return 0;
		}
	}

	// nnc --gemma-probe <file> [token_id] : run a partial forward pass
	// (embed -> attn_norm -> Q proj -> q_norm -> RoPE) on layer 0 of
	// a Gemma 3n model and print activation statistics. Smoke test for
	// the BF16 weights + kernels against real model data.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-probe") == 0 || strcmp(a, "-gemma-probe") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --gemma-probe requires a file path\n");
				return 1;
			}
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;
			int tok = gf.bos_id;
			if (i + 2 < argc && !parse_int_strict(argv[i + 2], &tok))
			{
				fprintf(stderr, "error: invalid token id '%s'\n", argv[i + 2]);
				gemma_free(gf);
				return 1;
			}
			const int rc = gemma_probe(gf, tok);
			gemma_free(gf);
			return rc;
		}
	}

	// nnc --gemma-forward <file> [token_id] : run the full single-token
	// forward pass through every layer (attention is degenerate at
	// pos=0) and print the top-5 next-token logits. End-to-end pipeline
	// smoke test against real Gemma weights.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-forward") == 0 || strcmp(a, "-gemma-forward") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "error: --gemma-forward requires a file path\n");
				return 1;
			}
			nnc_time_init();
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;
			int tok = gf.bos_id;
			if (i + 2 < argc && !parse_int_strict(argv[i + 2], &tok))
			{
				fprintf(stderr, "error: invalid token id '%s'\n", argv[i + 2]);
				gemma_free(gf);
				return 1;
			}
			const int rc = gemma_forward_one(gf, tok);
			gemma_free(gf);
			return rc;
		}
	}

	// nnc --gemma-gen <file> "id1 id2 id3 ..." [n_predict] [n_ctx]
	// : greedy generation. Token ids are space-separated decimal ints
	// (until we have a tokenizer). n_predict defaults to 16, n_ctx to
	// max(256, prompt+n_predict).
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-gen") == 0 || strcmp(a, "-gemma-gen") == 0)
		{
			if (i + 2 >= argc)
			{
				fprintf(stderr, "error: --gemma-gen requires <file> \"id1 id2 ...\" [n_predict] [n_ctx]\n");
				return 1;
			}
			nnc_time_init();
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;

			std::vector<int> prompt;
			{
				const char* s = argv[i + 2];
				char buf[32];
				size_t bp = 0;
				bool truncated = false;
				auto flush = [&]()
				{
					if (bp == 0) return;
					buf[bp] = 0;
					errno = 0;
					char* end = nullptr;
					const long v = strtol(buf, &end, 10);
					if (errno != 0 || end == buf || *end != '\0' || v < 0 || v > INT_MAX
						|| v >= static_cast<long>(gf.hparams.n_vocab))
					{
						fprintf(stderr, "error: invalid token id '%s' (vocab=%d)\n",
						        buf, gf.hparams.n_vocab);
						prompt.clear();
						prompt.push_back(-1); // sentinel: parsing failed
					}
					else
					{
						prompt.push_back(static_cast<int>(v));
					}
					bp = 0;
				};
				for (const char* p = s; *p; ++p)
				{
					if (*p == ' ' || *p == ',' || *p == '\t') flush();
					else if (bp + 1 < sizeof(buf)) buf[bp++] = *p;
					else truncated = true;
				}
				flush();
				if (truncated)
				{
					fprintf(stderr, "error: token id too long (>%zu chars)\n", sizeof(buf) - 1);
					gemma_free(gf);
					return 1;
				}
				if (!prompt.empty() && prompt.back() == -1)
				{
					gemma_free(gf);
					return 1;
				}
			}
			if (prompt.empty()) prompt.push_back(gf.bos_id);

			int n_predict = 16;
			if (i + 3 < argc && !parse_int_strict(argv[i + 3], &n_predict))
			{
				fprintf(stderr, "error: invalid n_predict '%s'\n", argv[i + 3]);
				gemma_free(gf);
				return 1;
			}
			int n_ctx = 0;
			if (i + 4 < argc && !parse_int_strict(argv[i + 4], &n_ctx))
			{
				fprintf(stderr, "error: invalid n_ctx '%s'\n", argv[i + 4]);
				gemma_free(gf);
				return 1;
			}
			if (n_ctx <= 0)
			{
				const int needed = static_cast<int>(prompt.size()) + n_predict;
				n_ctx = needed > 256 ? needed : 256;
			}
			const int rc = gemma_generate(gf, prompt, n_predict, n_ctx);
			gemma_free(gf);
			return rc;
		}
	}

	// nnc --gemma-tokenize <file> "text..." : run BPE on the prompt and
	// print the resulting token id / piece string list. Quick smoke test
	// for the tokenizer without spinning up the model forward pass.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-tokenize") == 0 || strcmp(a, "-gemma-tokenize") == 0)
		{
			if (i + 2 >= argc)
			{
				fprintf(stderr, "error: --gemma-tokenize requires <file> \"text\"\n");
				return 1;
			}
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;
			const std::vector<int> toks = gemma_tokenize(gf, argv[i + 2], gf.add_bos_token);
			printf("input    : %s\n", argv[i + 2]);
			printf("tokens   : %zu\n", toks.size());
			for (size_t k = 0; k < toks.size(); ++k)
			{
				const int t = toks[k];
				const char* s = (t >= 0 && t < static_cast<int>(gf.vocab_tokens.size()))
					                ? gf.vocab_tokens[t].c_str()
					                : "?";
				const float sc = (t >= 0 && t < static_cast<int>(gf.vocab_scores.size()))
					                 ? gf.vocab_scores[t]
					                 : 0.0f;
				printf("  [%3zu] id=%-7d score=%-9.4f '%s'\n", k, t, sc, s);
			}
			const std::string round = gemma_detokenize(gf, toks);
			printf("detok    : %s\n", round.c_str());
			// Quick lookup of common pieces for tokenizer sanity:
			for (const char* probe : {"\xE2\x96\x81" "capital", "\xE2\x96\x81" "is",
			                          "\xE2\x96\x81" "of", "\xE2\x96\x81" "The"})
			{
				auto it = gf.token_to_id.find(probe);
				if (it != gf.token_to_id.end())
				{
					const int id = it->second;
					const float sc = (id < (int)gf.vocab_scores.size())
						                 ? gf.vocab_scores[id] : 0.0f;
					printf("vocab '%s' -> id=%d score=%g\n", probe, id, sc);
				}
				else
				{
					printf("vocab '%s' -> NOT FOUND\n", probe);
				}
			}
			gemma_free(gf);
			return 0;
		}
	}

	// nnc --gemma-prompt <file> "text" [n_predict] [n_ctx]
	// : tokenize prompt, prefill, greedy-decode n_predict tokens.
	for (int i = 1; i < argc; i++)
	{
		const char* a = argv[i];
		if (strcmp(a, "--gemma-prompt") == 0 || strcmp(a, "-gemma-prompt") == 0)
		{
			if (i + 2 >= argc)
			{
				fprintf(stderr, "error: --gemma-prompt requires <file> \"text\" [n_predict] [n_ctx]\n");
				return 1;
			}
			nnc_time_init();
			gemma_file gf{};
			if (!gemma_load(argv[i + 1], gf)) return 1;
			if (!maybe_quantize(gf)) return 1;
			const std::vector<int> prompt = gemma_tokenize(gf, argv[i + 2], gf.add_bos_token);
			int n_predict = 16;
			if (i + 3 < argc && !parse_int_strict(argv[i + 3], &n_predict))
			{
				fprintf(stderr, "error: invalid n_predict '%s'\n", argv[i + 3]);
				gemma_free(gf);
				return 1;
			}
			int n_ctx = 0;
			if (i + 4 < argc && !parse_int_strict(argv[i + 4], &n_ctx))
			{
				fprintf(stderr, "error: invalid n_ctx '%s'\n", argv[i + 4]);
				gemma_free(gf);
				return 1;
			}
			if (n_ctx <= 0)
			{
				const int needed = static_cast<int>(prompt.size()) + n_predict;
				n_ctx = needed > 256 ? needed : 256;
			}
			const int rc = gemma_generate(gf, prompt, n_predict, n_ctx);
			gemma_free(gf);
			return rc;
		}
	}

	nnc_time_init();
	const int64_t t_main_start_us = nnc_time_us();

	gpt_params params;
	params.model = "models/gemma-4-E2B-it-BF16.gguf";

	if (gpt_params_parse(argc, argv, params) == false)
	{
		return 1;
	}

	if (params.seed < 0)
	{
		params.seed = static_cast<int32_t>(time(nullptr));
	}

	printf("%s: seed = %d\n", __func__, params.seed);

	// Interactive Gemma chat REPL: load the GGUF model once, then loop
	// reading prompts from stdin and streaming the generated text.
	// Ctrl-C exits.
	{
		// If the user didn't pass an explicit -m, scan the standard
		// model directories and ask which one to load.
		if (!model_explicit)
		{
			const std::vector<std::string> cands = collect_model_candidates();
			if (cands.empty())
			{
				fprintf(stderr,
				        "nnc: no .gguf models found under ./models, "
				        "$HOME/.lmstudio/models, or $HOME/models.\n"
				        "nnc: pass -m <file.gguf> to specify one.\n");
				return 1;
			}
			const std::string picked = prompt_for_model(cands);
			if (picked.empty())
			{
				fprintf(stderr, "nnc: no model selected.\n");
				return 1;
			}
			params.model = picked;
		}

		printf("nnc: loading %s\n", params.model.c_str());
		printf("nnc: press Ctrl-C to exit.\n");
		fflush(stdout);

		gemma_file gf{};
		if (!gemma_load(params.model.c_str(), gf))
		{
			fprintf(stderr, "%s: failed to load gemma model from '%s'\n",
			        __func__, params.model.c_str());
			return 1;
		}
		if (!maybe_quantize(gf)) return 1;

		const auto& h = gf.hparams;
		const int n_predict = params.n_predict > 0 ? params.n_predict : 256;

		// Look up Gemma chat-template special tokens by piece string.
		// Without these, the BPE merges split "<start_of_turn>" into
		// individual characters and the instruct model never sees the
		// real turn boundaries (and so emits one-token replies).
		// Gemma's GGUF stores control tokens with mangled piece
		// strings ("<|turn>" / "<turn|>") — try a few candidates.
		auto find_token = [&](const std::initializer_list<const char*> pieces) -> int
		{
			for (const char* p : pieces)
				for (size_t i = 0; i < gf.vocab_tokens.size(); ++i)
					if (gf.vocab_tokens[i] == p) return static_cast<int>(i);
			return -1;
		};
		// Validate a fallback id is in range and (if present) actually has
		// a vocab piece -- guards against bogus hard-coded ids on a model
		// with a smaller / different vocab.
		auto valid_id = [&](const int id) -> bool
		{
			return id >= 0 && static_cast<size_t>(id) < gf.vocab_tokens.size();
		};
		int tok_sot = find_token({"<start_of_turn>", "<|turn>"});
		int tok_eot = find_token({"<end_of_turn>", "<turn|>"});
		const int tok_nl = find_token({"\n"});
		// Gemma 3n GGUFs consistently place these control tokens at ids
		// 105/106. If the piece-string lookup misses (e.g. mangled in a
		// way we don't recognise), fall back to those ids when valid.
		if (tok_sot < 0 && valid_id(105)) tok_sot = 105;
		if (tok_eot < 0 && valid_id(106)) tok_eot = 106;
		// Tokenize the literal words "user" / "model" (no BOS) so we
		// don't depend on their absolute ids.
		std::vector<int> tok_user_word = gemma_tokenize(gf, "user", false);
		std::vector<int> tok_model_word = gemma_tokenize(gf, "model", false);

		const bool have_template = tok_sot >= 0 && tok_eot >= 0 && tok_nl >= 0;

		// Allocate the KV cache once for the entire REPL session and
		// keep appending across turns so the model actually sees the
		// conversation history. Use a fixed context window; on overflow
		// we wipe the cache and warn the user. Type `/reset` (or
		// `/clear`) at the prompt to start a fresh conversation, `/exit`
		// (or `/quit`) to leave.
		constexpr int n_ctx = 4096;
		gemma_kv_cache cache;
		if (!gemma_kv_init(gf, cache, n_ctx))
		{
			fprintf(stderr, "nnc: failed to init kv cache\n");
			gemma_free(gf);
			return 1;
		}

		std::string line;
		while (true)
		{
			printf("\n> ");
			fflush(stdout);
			if (!std::getline(std::cin, line)) break; // EOF (e.g. Ctrl-Z)
			if (line.empty()) continue;
			if (line == "/exit" || line == "/quit") break;
			if (line == "/reset" || line == "/clear")
			{
				cache.cur_pos = 0;
				printf("nnc: kv cache reset (0/%d)\n", n_ctx);
				continue;
			}

			// Build prompt token sequence. For instruct-tuned Gemma we
			// wrap in <start_of_turn>user\n ... <end_of_turn>\n
			// <start_of_turn>model\n; otherwise fall back to raw text.
			// Only emit BOS on the very first turn — subsequent turns
			// just append to the existing conversation.
			std::vector<int> prompt;
			if (have_template)
			{
				if (gf.bos_id >= 0 && cache.cur_pos == 0) prompt.push_back(gf.bos_id);
				prompt.push_back(tok_sot);
				prompt.insert(prompt.end(), tok_user_word.begin(), tok_user_word.end());
				prompt.push_back(tok_nl);
				{
					const std::vector<int> body = gemma_tokenize(gf, line, false);
					prompt.insert(prompt.end(), body.begin(), body.end());
				}
				prompt.push_back(tok_eot);
				prompt.push_back(tok_nl);
				prompt.push_back(tok_sot);
				prompt.insert(prompt.end(), tok_model_word.begin(), tok_model_word.end());
				prompt.push_back(tok_nl);
			}
			else
			{
				prompt = gemma_tokenize(gf, line,
				                        gf.add_bos_token && cache.cur_pos == 0);
			}
			if (prompt.empty()) continue;

			// If this turn won't fit, wipe the cache and start fresh
			// (with BOS re-added if the template needs it).
			const int needed = static_cast<int>(prompt.size()) + n_predict;
			if (cache.cur_pos + needed > n_ctx)
			{
				printf("nnc: context full (%d + %d > %d), resetting kv cache\n",
				       cache.cur_pos, needed, n_ctx);
				cache.cur_pos = 0;
				if (have_template && gf.bos_id >= 0 && prompt.front() != gf.bos_id)
					prompt.insert(prompt.begin(), gf.bos_id);
				if (cache.cur_pos + static_cast<int>(prompt.size()) + n_predict > n_ctx)
				{
					fprintf(stderr, "nnc: prompt too large for n_ctx=%d, skipping\n", n_ctx);
					continue;
				}
			}

			// Prefill prompt tokens (no output) starting at the current
			// cache position. Use the argmax-only path: it never
			// materialises the n_vocab-sized logits buffer (saves ~1 MB
			// of writes per token at vocab=262144) and skips softcap.
			const int base_pos = cache.cur_pos;
			int next = -1;
			bool ok = true;
			for (size_t i = 0; i < prompt.size(); ++i)
			{
				if (gemma_eval_token_argmax(gf, cache, prompt[i],
				                            base_pos + static_cast<int>(i),
				                            &next) != 0)
				{
					ok = false;
					break;
				}
			}
			if (!ok) continue;

			// Greedy decode, streaming detokenized text to stdout. The
			// generated tokens are appended to the same cache so the
			// next turn sees them as context. Flush only at newlines or
			// every N tokens — per-token fflush() forces a syscall and
			// blocks the decode loop on console drain.
			std::vector<int> one(1);
			constexpr int flush_every = 8;
			int since_flush = 0;
			for (int g = 0; g < n_predict; ++g)
			{
				if (next == gf.eos_id) break;
				one[0] = next;
				const std::string piece = gemma_detokenize(gf, one);
				fputs(piece.c_str(), stdout);
				if (++since_flush >= flush_every
					|| piece.find('\n') != std::string::npos)
				{
					fflush(stdout);
					since_flush = 0;
				}
				if (g + 1 >= n_predict) break;
				if (cache.cur_pos >= n_ctx) break;
				if (gemma_eval_token_argmax(gf, cache, next, cache.cur_pos, &next) != 0)
					break;
			}
			fflush(stdout);
			printf("\n");
		}

		gemma_kv_free(cache);
		gemma_free(gf);
		return 0;
	}
}
