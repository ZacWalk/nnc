// nnc — CLI argument parsing shared by main.cpp.

#pragma once

#include <string>

//
// CLI argument parsing
//

struct gpt_params
{
	int32_t seed = -1; // RNG seed
	int32_t n_predict = 200; // new tokens to predict

	// sampling parameters
	int32_t top_k = 40;
	float top_p = 0.9f;
	float temp = 1.0f;

	int32_t n_batch = 8; // batch size for prompt processing

	std::string model = "models/gemma-4-E2B-it-BF16.gguf"; // model path
	std::string prompt;
};

bool gpt_params_parse(int argc, char** argv, gpt_params& params);

void gpt_print_usage(int argc, char** argv, const gpt_params& params);
