// nnc — CLI argument parsing for the Gemma inference driver.

#include "utils.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace
{
	// Fetch the next argv slot for a flag that takes a value. Returns
	// nullptr (and prints to stderr) when the flag was the final argv
	// entry — without this guard, `argv[++i]` would read past argc.
	const char* next_value(const int argc, char** argv, int& i, const char* flag)
	{
		if (i + 1 >= argc)
		{
			fprintf(stderr, "error: %s requires a value\n", flag);
			return nullptr;
		}
		return argv[++i];
	}

	bool parse_int(const char* s, const char* flag, int& out)
	{
		try
		{
			size_t pos = 0;
			const long v = std::stol(s, &pos);
			if (pos == 0 || s[pos] != '\0')
			{
				fprintf(stderr, "error: invalid integer for %s: '%s'\n", flag, s);
				return false;
			}
			if (v < INT_MIN || v > INT_MAX)
			{
				fprintf(stderr, "error: integer out of range for %s: '%s'\n", flag, s);
				return false;
			}
			out = static_cast<int>(v);
			return true;
		}
		catch (const std::exception&)
		{
			fprintf(stderr, "error: invalid integer for %s: '%s'\n", flag, s);
			return false;
		}
	}

	bool parse_float(const char* s, const char* flag, float& out)
	{
		try
		{
			size_t pos = 0;
			const float v = std::stof(s, &pos);
			if (pos == 0 || s[pos] != '\0')
			{
				fprintf(stderr, "error: invalid float for %s: '%s'\n", flag, s);
				return false;
			}
			out = v;
			return true;
		}
		catch (const std::exception&)
		{
			fprintf(stderr, "error: invalid float for %s: '%s'\n", flag, s);
			return false;
		}
	}
}

bool gpt_params_parse(const int argc, char** argv, gpt_params& params)
{
	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];
		const char* flag = argv[i];

		if (arg == "-s" || arg == "--seed")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v) return false;
			int seed = 0;
			if (!parse_int(v, flag, seed)) return false;
			params.seed = seed;
		}
		else if (arg == "-p" || arg == "--prompt")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v) return false;
			params.prompt = v;
		}
		else if (arg == "-n" || arg == "--n_predict")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v || !parse_int(v, flag, params.n_predict)) return false;
		}
		else if (arg == "--top_k")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v || !parse_int(v, flag, params.top_k)) return false;
		}
		else if (arg == "--top_p")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v || !parse_float(v, flag, params.top_p)) return false;
		}
		else if (arg == "--temp")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v || !parse_float(v, flag, params.temp)) return false;
		}
		else if (arg == "-b" || arg == "--batch_size")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v || !parse_int(v, flag, params.n_batch)) return false;
		}
		else if (arg == "-m" || arg == "--model")
		{
			const char* v = next_value(argc, argv, i, flag);
			if (!v) return false;
			params.model = v;
		}
		else if (arg == "-h" || arg == "--help")
		{
			gpt_print_usage(argc, argv, params);
			exit(0);
		}
		else
		{
			fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
			gpt_print_usage(argc, argv, params);
			return false;
		}
	}

	return true;
}

void gpt_print_usage(int argc, char** argv, const gpt_params& params)
{
	fprintf(stderr, "usage: %s [options]\n", argv[0]);
	fprintf(stderr, "\n");
	fprintf(stderr, "options:\n");
	fprintf(stderr, "  -h, --help            show this help message and exit\n");
	fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
	fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
	fprintf(stderr, "                        prompt to start generation with (default: random)\n");
	fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
	fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
	fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
	fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
	fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
	fprintf(stderr, "  -m FNAME, --model FNAME\n");
	fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
	fprintf(stderr, "\n");
}
