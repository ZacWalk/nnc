// nnc — CLI entry point and Gemma inference driver.
// Parses arguments, dispatches to --test and the various --gguf-*/--gemma-*
// inspection modes, otherwise loads a Gemma GGUF model and runs an
// interactive chat REPL against the nnc runtime.

#include "runtime.h"

#include "gemma.h"
#include "gguf.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

int run_tests(); // tests.cpp

int main(int argc, char** argv)
{
#ifdef _DEBUG
	// Route assert failures to stderr instead of a modal dialog so the
	// process actually exits when something is wrong.
	_set_error_mode(_OUT_TO_STDERR);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
	// Force UTF-8 on the Windows console so tokens containing U+2581 (▁),
	// em-dashes, etc. render correctly instead of as cp437 mojibake.
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
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
			const int tok = (i + 2 < argc) ? atoi(argv[i + 2]) : gf.bos_id;
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
			const int tok = (i + 2 < argc) ? atoi(argv[i + 2]) : gf.bos_id;
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

			std::vector<int> prompt;
			{
				const char* s = argv[i + 2];
				char buf[32];
				size_t bp = 0;
				auto flush = [&]()
				{
					if (bp == 0) return;
					buf[bp] = 0;
					prompt.push_back(atoi(buf));
					bp = 0;
				};
				for (const char* p = s; *p; ++p)
				{
					if (*p == ' ' || *p == ',' || *p == '\t') flush();
					else if (bp + 1 < sizeof(buf)) buf[bp++] = *p;
				}
				flush();
			}
			if (prompt.empty()) prompt.push_back(gf.bos_id);

			const int n_predict = (i + 3 < argc) ? atoi(argv[i + 3]) : 16;
			int n_ctx = (i + 4 < argc) ? atoi(argv[i + 4]) : 0;
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
			const std::vector<int> toks = gemma_tokenize(gf, argv[i + 2], gf.add_bos_token);
			printf("input    : %s\n", argv[i + 2]);
			printf("tokens   : %zu\n", toks.size());
			for (size_t k = 0; k < toks.size(); ++k)
			{
				const int t = toks[k];
				const char* s = (t >= 0 && t < static_cast<int>(gf.vocab_tokens.size()))
					                ? gf.vocab_tokens[t].c_str()
					                : "?";
				printf("  [%3zu] id=%-7d '%s'\n", k, t, s);
			}
			const std::string round = gemma_detokenize(gf, toks);
			printf("detok    : %s\n", round.c_str());
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
			const std::vector<int> prompt = gemma_tokenize(gf, argv[i + 2], gf.add_bos_token);
			const int n_predict = (i + 3 < argc) ? atoi(argv[i + 3]) : 16;
			int n_ctx = (i + 4 < argc) ? atoi(argv[i + 4]) : 0;
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
	params.model = "models\\gemma-4-E2B-it-BF16.gguf";

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
		const int tok_sot = find_token({"<start_of_turn>", "<|turn>"});
		const int tok_eot = find_token({"<end_of_turn>", "<turn|>"});
		const int tok_nl = find_token({"\n"});
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
		const int n_ctx = 4096;
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
			// next turn sees them as context.
			for (int g = 0; g < n_predict; ++g)
			{
				if (next == gf.eos_id) break;
				const std::string piece = gemma_detokenize(gf, {next});
				fputs(piece.c_str(), stdout);
				fflush(stdout);
				if (g + 1 >= n_predict) break;
				if (cache.cur_pos >= n_ctx) break;
				if (gemma_eval_token_argmax(gf, cache, next, cache.cur_pos, &next) != 0)
					break;
			}
			printf("\n");
		}

		gemma_kv_free(cache);
		gemma_free(gf);
		return 0;
	}
}
