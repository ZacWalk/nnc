// nnc — CLI argument parsing for the Gemma inference driver.

#include "utils.h"

#include <cstdio>
#include <cstdlib>

bool gpt_params_parse(const int argc, char** argv, gpt_params& params)
{
	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];

		if (arg == "-s" || arg == "--seed")
		{
			params.seed = std::stoi(argv[++i]);
		}
		else if (arg == "-p" || arg == "--prompt")
		{
			params.prompt = argv[++i];
		}
		else if (arg == "-n" || arg == "--n_predict")
		{
			params.n_predict = std::stoi(argv[++i]);
		}
		else if (arg == "--top_k")
		{
			params.top_k = std::stoi(argv[++i]);
		}
		else if (arg == "--top_p")
		{
			params.top_p = std::stof(argv[++i]);
		}
		else if (arg == "--temp")
		{
			params.temp = std::stof(argv[++i]);
		}
		else if (arg == "-b" || arg == "--batch_size")
		{
			params.n_batch = std::stoi(argv[++i]);
		}
		else if (arg == "-m" || arg == "--model")
		{
			params.model = argv[++i];
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
			exit(0);
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
