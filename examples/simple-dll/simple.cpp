#include "common.h"
#include "llama.h"

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

llama_batch batch = llama_batch_init(512, 0, 1);

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

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}

char *prompt(std::string prompt)
{
    std::vector<char> res;

    llama_batch_clear(batch);

    // total length of the sequence including the prompt
    const int n_predict = params.n_predict;

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    auto n_past = common_part(cache_tokens, tokens_list);

    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

    cache_tokens = tokens_list;

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx)
    {
        return " ";
    }

    // evaluate the initial prompt
    for (size_t i = n_past; i < tokens_list.size(); i++)
    {
        llama_batch_add(batch, tokens_list[i], i, {0}, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        return " ";
    }

    // main loop

    int n_cur = batch.n_tokens;
    int n_decode = 0;

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

            cache_tokens.push_back(new_token_id);

            auto token_piece = llama_token_to_piece(ctx, new_token_id);

            for (auto c : token_piece)
            {
                res.push_back(c);
            }

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict)
            {
                break;
            }

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
            return " ";
        }
    }

    llama_batch_free(batch);
}

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }

    return i;
}