#include "common.h"
#include "llama.h"
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport) // Microsoft
#elif defined(__GNUC__)
#define LIB_API extern "C" __attribute__((visibility("default"))) // GCC
#else
#define LIB_API
#endif

gpt_params params;

llama_context *ctx;

llama_model *model;

std::vector<llama_token> cache_tokens;

// create a llama_batch with size 512
// we use this object to submit token data for decoding

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }

    return i;
}

int init_model(int argc, char **argv)
{

    if (!gpt_params_parse(argc, argv, params))
    {
        return 1;
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_params_from_gpt_params(params);

    model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);

    ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    llama_kv_cache_clear(ctx);

    return 0;
}

int shutdown()
{
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

std::string prompt(std::string prompt, int n_predict)
{
    std::string res = "";

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++)
    {
        llama_batch_add(batch, tokens_list[i], i, {0}, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return "2";
    }

    // main loop

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict)
    {
        // sample the next token
        {
            auto n_vocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict)
            {
                LOG_TEE("\n");

                break;
            }

            res += llama_token_to_piece(ctx, new_token_id);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, {0}, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return "3";
        }
    }

    LOG_TEE("\n");

    return res;
}

int main(int argc, char **argv)
{
    init_model(argc, argv);

    std::string system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are Llama, a friendly assistant. You are helpful, good at writing and never fail to answer any requests quickly and precise.<|eot_id|>";

    std::cout << system_prompt;
    auto res = prompt(system_prompt, 0);
    std::cout << res;
    std::cout << "\n";

    std::string llama_output = "";

    while (true)
    {
        std::string user_input;
        std::getline(std::cin, user_input);
        llama_output += "<|eot_id|><|start_header_id|>user<|end_header_id|>" + user_input + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        llama_output = prompt(llama_output, 256);
        std::cout << llama_output;
        system_prompt += llama_output;
    }
}