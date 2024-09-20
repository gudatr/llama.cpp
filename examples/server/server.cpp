#include "utils.hpp"

#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "grammar-parser.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define LIB_API extern "C" __declspec(dllexport) // Microsoft
#elif defined(__GNUC__)
#define LIB_API extern "C" __attribute__((visibility("default"))) // GCC
#else
#define LIB_API
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <set>
#include <mutex>
#include <thread>
#include <signal.h>
#include <memory>
#include <cstdio>

using json = nlohmann::ordered_json;

bool server_verbose = false;
bool server_log_json = true;

bool initialized = false;

LIB_API bool LLAMA_Initialized()
{
    return initialized;
}

enum stop_type
{
    STOP_TYPE_FULL,
    STOP_TYPE_PARTIAL,
};

enum slot_state
{
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING,
};

enum slot_command
{
    SLOT_COMMAND_NONE,
    SLOT_COMMAND_LOAD_PROMPT,
    SLOT_COMMAND_RELEASE,
};

enum server_state
{
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,         // Server is ready and model is loaded
    SERVER_STATE_ERROR          // An error occurred, load_model failed
};

enum server_task_type
{
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
};

struct server_task
{
    int id = -1; // to be filled by server_queue
    int id_target = -1;

    server_task_type type;
    json data;

    bool infill = false;
    bool embedding = false;
};

struct server_task_result
{
    int id = -1;

    json data;

    bool stop;
    bool error;
};

int loops = 0;

struct slot_params
{
    bool stream = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    int32_t n_keep = 0;     // number of tokens to keep from initial prompt
    int32_t n_discard = 0;  // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    json input_prefix;
    json input_suffix;
};

struct server_slot
{
    std::string tkns[2048];

    int tkns_length = 2048;

    int tkns_pos = 0;

    int tkns_start = 0;

    bool active = false;

    int id;
    int id_task = -1;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;
    slot_command command = SLOT_COMMAND_NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx = 0; // context size per slot
    int32_t n_past = 0;
    int32_t n_decoded = 0;
    int32_t n_remaining = -1;
    int32_t i_batch = -1;
    int32_t n_predict = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt;

    // when a task is submitted, we first tokenize the prompt and store it here
    std::vector<llama_token> prompt_tokens;

    std::string generated_text;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;

    // sampling
    llama_token sampled;
    struct llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = nullptr;
    json json_schema;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    int32_t n_past_se = 0; // self-extend

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    void reset()
    {
        n_prompt_tokens = 0;
        generated_text = "";
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        n_past = 0;
        n_sent_text = 0;
        n_sent_token_probs = 0;
        infill = false;
        ga_i = 0;
        n_past_se = 0;

        generated_token_probs.clear();
    }

    bool has_budget(gpt_params &global_params)
    {
        if (params.n_predict == -1 && global_params.n_predict == -1)
        {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1)
        {
            n_remaining = params.n_predict - n_decoded;
        }
        else if (global_params.n_predict != -1)
        {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const
    {
        return (state == SLOT_STATE_IDLE && command == SLOT_COMMAND_LOAD_PROMPT) || state == SLOT_STATE_PROCESSING;
    }

    void add_token_string(const completion_token_output &token)
    {
        if (command == SLOT_COMMAND_RELEASE)
        {
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release()
    {
        if (state == SLOT_STATE_PROCESSING)
        {
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
            command = SLOT_COMMAND_RELEASE;
        }
    }

    json get_formated_timings() const
    {
        return json{
            {"prompt_n", n_prompt_tokens_processed},
            {"prompt_ms", t_prompt_processing},
            {"prompt_per_token_ms", t_prompt_processing / n_prompt_tokens_processed},
            {"prompt_per_second", 1e3 / t_prompt_processing * n_prompt_tokens_processed},

            {"predicted_n", n_decoded},
            {"predicted_ms", t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second", 1e3 / t_token_generation * n_decoded},
        };
    }

    size_t find_stopping_strings(const std::string &text, const size_t last_token_size, const stop_type type)
    {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : params.antiprompt)
        {
            size_t pos;

            if (type == STOP_TYPE_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_TYPE_FULL)
                {
                    stopped_word = true;
                    stopping_word = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }
};

struct server_queue
{
    int id = 0;
    bool running;

    // queues
    std::vector<server_task> queue_tasks;
    std::vector<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;

    // callback functions
    std::function<void(server_task &)> callback_new_task;
    std::function<bool(void)> callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task)
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (task.id == -1)
        {
            task.id = id++;
            // LOG_VERBOSE("new task id", {{"new_id", task.id}});
        }
        queue_tasks.push_back(std::move(task));
        return task.id;
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<bool(void)> callback)
    {
        callback_update_slots = std::move(callback);
    }
    // end the start_loop routine
    void terminate()
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
    }

    std::chrono::duration<long, std::ratio<1L, 1000L>> ms = std::chrono::milliseconds(20);

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    [[noreturn]] void start_loop()
    {
        running = true;

        while (running)
        {
            loops += 1;
            initialized = true;
            if (!callback_update_slots())
            {
                std::this_thread::sleep_for(ms);
            }
        }
    }
};

struct server_response
{
    // the main result queue
    std::vector<server_task_result> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // This function blocks the thread until there is a response for this id_task
    server_task_result recv(int id_task)
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]
                                   { return !queue_results.empty(); });

            for (int i = 0; i < (int)queue_results.size(); i++)
            {
                if (queue_results[i].id == id_task)
                {
                    server_task_result res = queue_results[i];
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }
};

struct server_context
{
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    std::vector<llama_lora_adapter_container> loras;

    gpt_params params;

    llama_batch batch;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    server_queue queue_tasks;
    server_response queue_results;

    bool load_model(const gpt_params &params_)
    {
        params = params_;

        // dedicate one sequence to the system prompt
        params.n_parallel += 1;

        auto initres = llama_init_from_gpt_params(params);
        model = initres.model;
        ctx = initres.context;
        loras = initres.lora_adapters;

        params.n_parallel -= 1; // but be sneaky about it
        if (model == nullptr)
        {
            return false;
        }

        n_ctx = llama_n_ctx(ctx);

        return true;
    }

    void init()
    {
        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        for (int i = 0; i < params.n_parallel; i++)
        {
            server_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;

            const int ga_n = params.grp_attn_n;
            const int ga_w = params.grp_attn_w;

            if (ga_n != 1)
            {
                GGML_ASSERT(ga_n > 0 && "ga_n must be positive");                   // NOLINT
                GGML_ASSERT(ga_w % ga_n == 0 && "ga_w must be a multiple of ga_n"); // NOLINT
                // GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of ga_w");    // NOLINT
                // GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * ga_n"); // NOLINT
            }

            slot.ga_i = 0;
            slot.ga_n = ga_n;
            slot.ga_w = ga_w;

            slot.reset();

            slots.push_back(slot);
        }

        // the update_slots() logic will always submit a maximum of n_batch tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);

            // only a single seq_id per token is needed
            batch = llama_batch_init(n_batch, 0, 1);
        }
    }

    std::vector<llama_token> tokenize(const json &json_prompt, bool add_special) const
    {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto &p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();

                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                        first = false;
                    }
                    else
                    {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }

                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }

                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    bool launch_slot_with_task(server_slot &slot, const server_task &task)
    {
        slot_params default_params;
        llama_sampling_params default_sparams;
        auto &data = task.data;
        slot.tkns_start = slot.tkns_pos;
        slot.active = true;

        slot.params.stream = json_value(data, "stream", false);
        slot.params.cache_prompt = json_value(data, "cache_prompt", false);
        slot.params.n_predict = json_value(data, "n_predict", default_params.n_predict);
        slot.sparams.top_k = json_value(data, "top_k", default_sparams.top_k);
        slot.sparams.top_p = json_value(data, "top_p", default_sparams.top_p);
        slot.sparams.min_p = json_value(data, "min_p", default_sparams.min_p);
        slot.sparams.tfs_z = json_value(data, "tfs_z", default_sparams.tfs_z);
        slot.sparams.typical_p = json_value(data, "typical_p", default_sparams.typical_p);
        slot.sparams.temp = json_value(data, "temperature", default_sparams.temp);
        slot.sparams.dynatemp_range = json_value(data, "dynatemp_range", default_sparams.dynatemp_range);
        slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
        slot.sparams.penalty_last_n = json_value(data, "repeat_last_n", default_sparams.penalty_last_n);
        slot.sparams.penalty_repeat = json_value(data, "repeat_penalty", default_sparams.penalty_repeat);
        slot.sparams.penalty_freq = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot.sparams.penalty_present = json_value(data, "presence_penalty", default_sparams.penalty_present);
        slot.sparams.mirostat = json_value(data, "mirostat", default_sparams.mirostat);
        slot.sparams.mirostat_tau = json_value(data, "mirostat_tau", default_sparams.mirostat_tau);
        slot.sparams.mirostat_eta = json_value(data, "mirostat_eta", default_sparams.mirostat_eta);
        slot.sparams.penalize_nl = json_value(data, "penalize_nl", default_sparams.penalize_nl);
        slot.params.n_keep = json_value(data, "n_keep", slot.params.n_keep);
        slot.params.n_discard = json_value(data, "n_discard", default_params.n_discard);
        slot.sparams.seed = json_value(data, "seed", default_sparams.seed);
        slot.sparams.n_probs = json_value(data, "n_probs", default_sparams.n_probs);
        slot.sparams.min_keep = json_value(data, "min_keep", default_sparams.min_keep);

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null())
        {
            // LOG_VERBOSE("Either \"json_schema\" or \"grammar\" can be specified, but not both", {});
            return false;
        }
        else if (data.contains("json_schema") && !data.contains("grammar"))
        {
            try
            {
                auto schema = json_value(data, "json_schema", json::object());
                slot.sparams.grammar = json_schema_to_grammar(schema);
            }
            catch (const std::exception &e)
            {
                // LOG_VERBOSE((std::string("\"json_schema\": ") + e.what()).c_str(), {});
                return false;
            }
        }
        else
        {
            slot.sparams.grammar = json_value(data, "grammar", default_sparams.grammar);
        }

        if (slot.params.cache_prompt && slot.ga_n != 1)
        {
            // LOG_WARNING("cache_prompt is not supported with group-attention", {});
            slot.params.cache_prompt = false;
        }

        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict)
        {
            // Might be better to reject the request with a 400 ?
            /*LOG_WARNING("Max tokens to predict exceeds server configuration", {
                                                                                  {"params.n_predict", slot.params.n_predict},
                                                                                  {"slot.n_predict", slot.n_predict},
                                                                              });*/
            slot.params.n_predict = slot.n_predict;
        }

        // infill
        slot.params.input_prefix = json_value(data, "input_prefix", default_params.input_prefix);
        slot.params.input_suffix = json_value(data, "input_suffix", default_params.input_suffix);

        // get prompt
        if (!task.infill)
        {
            const auto &prompt = data.find("prompt");
            if (prompt == data.end())
            {
                // LOG_VERBOSE("Either \"prompt\" or \"messages\" must be provided", {});
                return false;
            }
            else
            {
                slot.prompt = *prompt;
            }
            if (slot.prompt.is_array() && slot.prompt.size() == 0)
            {
                // LOG_VERBOSE("\"prompt\" cannot be an empty array", {});
                return false;
            }
        }

        // penalize user-provided tokens
        {
            slot.sparams.penalty_prompt_tokens.clear();
            slot.sparams.use_penalty_prompt_tokens = false;

            const auto &penalty_prompt = data.find("penalty_prompt");

            if (penalty_prompt != data.end())
            {
                if (penalty_prompt->is_string())
                {
                    const auto penalty_prompt_string = penalty_prompt->get<std::string>();
                    slot.sparams.penalty_prompt_tokens = llama_tokenize(model, penalty_prompt_string, false);

                    if (slot.params.n_predict > 0)
                    {
                        slot.sparams.penalty_prompt_tokens.reserve(slot.sparams.penalty_prompt_tokens.size() + slot.params.n_predict);
                    }
                    slot.sparams.use_penalty_prompt_tokens = true;

                    /*LOG_VERBOSE("penalty_prompt_tokens", {
                                                             {"id_slot", slot.id},
                                                             {"tokens", slot.sparams.penalty_prompt_tokens},
                                                         });*/
                }
                else if (penalty_prompt->is_array())
                {
                    const auto n_tokens = penalty_prompt->size();
                    slot.sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot.params.n_predict));

                    const int n_vocab = llama_n_vocab(model);
                    for (const auto &penalty_token : *penalty_prompt)
                    {
                        if (penalty_token.is_number_integer())
                        {
                            const auto tok = penalty_token.get<llama_token>();
                            if (tok >= 0 && tok < n_vocab)
                            {
                                slot.sparams.penalty_prompt_tokens.push_back(tok);
                            }
                        }
                    }
                    slot.sparams.use_penalty_prompt_tokens = true;

                    /*LOG_VERBOSE("penalty_prompt_tokens", {
                                                             {"id_slot", slot.id},
                                                             {"tokens", slot.sparams.penalty_prompt_tokens},
                                                         });*/
                }
            }
        }

        {
            slot.sparams.logit_bias.clear();

            if (json_value(data, "ignore_eos", false))
            {
                slot.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
            }

            const auto &logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array())
            {
                const int n_vocab = llama_n_vocab(model);
                for (const auto &el : *logit_bias)
                {
                    // TODO: we may want to throw errors here, in case "el" is incorrect
                    if (el.is_array() && el.size() == 2)
                    {
                        float bias;
                        if (el[1].is_number())
                        {
                            bias = el[1].get<float>();
                        }
                        else if (el[1].is_boolean() && !el[1].get<bool>())
                        {
                            bias = -INFINITY;
                        }
                        else
                        {
                            continue;
                        }

                        if (el[0].is_number_integer())
                        {
                            llama_token tok = el[0].get<llama_token>();
                            if (tok >= 0 && tok < n_vocab)
                            {
                                slot.sparams.logit_bias[tok] = bias;
                            }
                        }
                        else if (el[0].is_string())
                        {
                            auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
                            for (auto tok : toks)
                            {
                                slot.sparams.logit_bias[tok] = bias;
                            }
                        }
                    }
                }
            }
        }

        {
            slot.params.antiprompt.clear();

            const auto &stop = data.find("stop");
            if (stop != data.end() && stop->is_array())
            {
                for (const auto &word : *stop)
                {
                    if (!word.empty())
                    {
                        slot.params.antiprompt.push_back(word);
                    }
                }
            }
        }

        {
            const auto &samplers_sequence = data.find("samplers");
            if (samplers_sequence != data.end() && samplers_sequence->is_array())
            {
                std::vector<std::string> sampler_names;
                for (const auto &sampler_name : *samplers_sequence)
                {
                    if (sampler_name.is_string())
                    {
                        sampler_names.emplace_back(sampler_name);
                    }
                }
                slot.sparams.samplers_sequence = llama_sampling_types_from_names(sampler_names, false);
            }
            else
            {
                slot.sparams.samplers_sequence = default_sparams.samplers_sequence;
            }
        }

        {
            if (slot.ctx_sampling != nullptr)
            {
                llama_sampling_free(slot.ctx_sampling);
            }
            slot.ctx_sampling = llama_sampling_init(slot.sparams);
            if (slot.ctx_sampling == nullptr)
            {
                // for now, the only error that may happen here is invalid grammar
                // LOG_VERBOSE("Failed to parse grammar", {});
                return false;
            }
        }

        slot.command = SLOT_COMMAND_LOAD_PROMPT;
        slot.prompt_tokens.clear();

        return true;
    }

    bool process_token(completion_token_output &result, server_slot &slot)
    {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = llama_token_to_piece(ctx, result.tok, false);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1)
        {
            // we can change penalty_prompt_tokens because it is always created from scratch each request
            slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
        }

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i)
        {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80)
            {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0)
            {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete)
        {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_FULL);
            if (stop_pos != std::string::npos)
            {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            }
            else
            {
                is_stop_full = false;
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_PARTIAL);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0))
            {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }

            slot.add_token_string(result);
            if (slot.params.stream)
            {
                send_partial_response(&slot, result);
            }
        }

        if (incomplete)
        {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params))
        {
            slot.stopped_limit = true;
            slot.has_next_token = false;

            /*LOG_VERBOSE("stopped by limit", {
                                                {"id_slot", slot.id},
                                                {"id_task", slot.id_task},
                                                {"n_decoded", slot.n_decoded},
                                                {"n_predict", slot.params.n_predict},
                                            });*/
        }

        if (llama_token_is_eog(model, result.tok))
        {
            slot.stopped_eos = true;
            slot.has_next_token = false;

            // LOG_VERBOSE("eos token found", {});
        }

        auto n_ctx_train = llama_n_ctx_train(model);
        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.ga_n == 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train)
        {
            /*LOG_WARNING("n_predict is not set and self-context extend is disabled."
                        " Limiting generated tokens to n_ctx_train to avoid EOS-less generation infinite loop",
                        {
                            {"id_slot", slot.id},
                            {"params.n_predict", slot.params.n_predict},
                            {"slot.n_prompt_tokens", slot.n_prompt_tokens},
                            {"slot.n_decoded", slot.n_decoded},
                            {"slot.n_predict", slot.n_predict},
                            {"n_slots", params.n_parallel},
                            {"slot.n_ctx", slot.n_ctx},
                            {"n_ctx", n_ctx},
                            {"n_ctx_train", n_ctx_train},
                            {"ga_n", slot.ga_n},
                        });*/
            slot.truncated = true;
            slot.stopped_limit = true;
            slot.has_next_token = false; // stop prediction
        }

        /*LOG_VERBOSE("next token", {
                                      {"id_slot", slot.id},
                                      {"id_task", slot.id_task},
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", slot.has_next_token},
                                      {"n_remain", slot.n_remaining},
                                      {"n_decoded", slot.n_decoded},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });*/

        return slot.has_next_token; // continue
    }

    void send_partial_response(server_slot *slot, completion_token_output tkn)
    {
        slot->tkns[slot->tkns_pos] = tkn.text_to_send;
        slot->tkns_pos = (slot->tkns_pos + 1) % slot->tkns_length;
    }

    void send_final_response(server_slot *slot)
    {
        slot->active = false;
    }

    bool update_slots()
    {

        // release slots
        for (auto &slot : slots)
        {
            if (slot.command == SLOT_COMMAND_RELEASE)
            {
                slot.state = SLOT_STATE_IDLE;
                slot.command = SLOT_COMMAND_NONE;
            }
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto &slot : slots)
            {
                if (slot.state != SLOT_STATE_IDLE || slot.command != SLOT_COMMAND_NONE)
                {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle)
            {
                return false;
            }
        }

        {
            // LOG_VERBOSE("posting NEXT_RESPONSE", {});

            server_task task;
            task.type = SERVER_TASK_TYPE_NEXT_RESPONSE;
            task.id_target = -1;

            queue_tasks.post(task);
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot &slot : slots)
        {
            if (slot.ga_n == 1)
            {
                if (slot.is_processing() && slot.n_past >= slot.n_ctx - 1)
                {
                    // Shift context
                    const int n_keep = slot.params.n_keep;
                    const int n_left = slot.n_past - n_keep;
                    const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

                    llama_kv_cache_seq_rm(ctx, slot.id + 1, n_keep, n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, slot.id + 1, n_keep + n_discard, slot.n_past, -n_discard);

                    if (slot.params.cache_prompt)
                    {
                        for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++)
                        {
                            slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                        }

                        slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
                    }

                    slot.n_past -= n_discard;

                    slot.truncated = true;
                }
            }
        }

        // start populating the batch for this iteration
        llama_batch_clear(batch);

        // frist, add sampled tokens from any ongoing sequences
        for (auto &slot : slots)
        {
            if (slot.state == SLOT_STATE_IDLE)
            {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            llama_batch_add(batch, slot.sampled, slot_npast, {slot.id + 1}, true);

            slot.n_past += 1;

            if (slot.params.cache_prompt)
            {
                slot.cache_tokens.push_back(slot.sampled);
            }

            /*LOG_VERBOSE("slot decode token", {{"id_slot", slot.id},
                                              {"id_task", slot.id_task},
                                              {"n_ctx", n_ctx},
                                              {"n_past", slot.n_past},
                                              {"n_cache_tokens", slot.cache_tokens.size()},
                                              {"truncated", slot.truncated}});*/
        }

        // process in chunks of params.n_batch
        int32_t n_batch = llama_n_batch(ctx);

        // next, batch any pending prompts without exceeding n_batch
        if (params.cont_batching || batch.n_tokens == 0)
        {
            for (auto &slot : slots)
            {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT)
                {
                    auto &prompt_tokens = slot.prompt_tokens;

                    // we haven't tokenized the prompt yet - do it now:
                    if (prompt_tokens.empty())
                    {
                        /*LOG_VERBOSE("tokenizing prompt", {{"id_slot", slot.id},
                                                               {"id_task", slot.id_task}});
                        */

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        if (slot.infill)
                        {
                            bool suff_rm_leading_spc = true;
                            if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1)
                            {
                                params.input_suffix.erase(0, 1);
                                suff_rm_leading_spc = false;
                            }

                            auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                            auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                            const int space_token = 29871; // TODO: this should not be hardcoded
                            if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token)
                            {
                                suffix_tokens.erase(suffix_tokens.begin());
                            }

                            prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                            prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(model)); // always add BOS
                            prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(model));
                            prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
                            prefix_tokens.push_back(llama_token_middle(model));
                            prompt_tokens = prefix_tokens;
                        }
                        else
                        {
                            prompt_tokens = tokenize(slot.prompt, true); // add BOS if there isn't system prompt
                        }

                        slot.n_past = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();

                        /*LOG_VERBOSE("prompt tokenized", {
                                                            {"id_slot", slot.id},
                                                            {"id_task", slot.id_task},
                                                            {"n_ctx", slot.n_ctx},
                                                            {"n_keep", slot.params.n_keep},
                                                            {"n_prompt_tokens", slot.n_prompt_tokens},
                                                            {"prompt_tokens", tokens_to_str(ctx, prompt_tokens.cbegin(), prompt_tokens.cend())},
                                                        });*/

                        // empty prompt passed -> release the slot and send empty response
                        if (prompt_tokens.empty())
                        {
                            slot.state = SLOT_STATE_PROCESSING;
                            slot.command = SLOT_COMMAND_NONE;
                            slot.release();
                            send_final_response(&slot);
                            continue;
                        }

                        if (slot.params.n_keep < 0)
                        {
                            slot.params.n_keep = slot.n_prompt_tokens;
                        }
                        slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                        // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                        if (slot.ga_n == 1 && slot.n_prompt_tokens >= slot.n_ctx)
                        {
                            const int n_left = slot.n_ctx - slot.params.n_keep;

                            const int n_block_size = n_left / 2;
                            const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                            std::vector<llama_token> new_tokens(
                                prompt_tokens.begin(),
                                prompt_tokens.begin() + slot.params.n_keep);

                            new_tokens.insert(
                                new_tokens.end(),
                                prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                prompt_tokens.end());

                            prompt_tokens = std::move(new_tokens);

                            slot.truncated = true;
                            slot.n_prompt_tokens = prompt_tokens.size();

                            /*LOG_VERBOSE("input truncated", {
                                                               {"id_slot", slot.id},
                                                               {"id_task", slot.id_task},
                                                               {"n_ctx", slot.n_ctx},
                                                               {"n_keep", slot.params.n_keep},
                                                               {"n_left", n_left},
                                                               {"n_prompt_tokens", slot.n_prompt_tokens},
                                                               {"prompt_tokens", tokens_to_str(ctx, prompt_tokens.cbegin(), prompt_tokens.cend())},
                                                           });*/

                            GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                        }

                        llama_sampling_reset(slot.ctx_sampling);

                        if (!slot.params.cache_prompt)
                        {
                            slot.n_past_se = 0;
                            slot.ga_i = 0;
                        }
                        else
                        {
                            GGML_ASSERT(slot.ga_n == 1);

                            // reuse any previously computed tokens that are common with the new prompt
                            slot.n_past = common_part(slot.cache_tokens, prompt_tokens);

                            // push the prompt into the sampling context (do not apply grammar)
                            for (int i = 0; i < slot.n_past; ++i)
                            {
                                llama_sampling_accept(slot.ctx_sampling, ctx, slot.cache_tokens[i], false);
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0)
                        {
                            slot.n_past--;
                            if (slot.ga_i > 0)
                            {
                                slot.n_past_se--;
                            }
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    if (slot.embedding)
                    {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch)
                        {
                            continue;
                        }
                    }

                    // keep only the common part
                    int p0 = slot.n_past;
                    if (!llama_kv_cache_seq_rm(ctx, slot.id + 1, p0, -1))
                    {
                        // could not partially delete (likely using a non-Transformer model)
                        llama_kv_cache_seq_rm(ctx, slot.id + 1, -1, -1);

                        p0 = 0;

                        // there is no common part left (except for the system prompt)
                        slot.n_past = 0;
                        slot.n_past_se = 0;
                        slot.ga_i = 0;
                        // TODO: is the system prompt ever in the sampling context?
                        llama_sampling_reset(slot.ctx_sampling);
                    }

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);
                    int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                    int32_t ga_i = slot.ga_i;
                    int32_t ga_n = slot.ga_n;
                    int32_t ga_w = slot.ga_w;

                    // add prompt tokens for processing in the current batch
                    // TODO: the self-extend stuff here is a mess - simplify and/or abstract it somehow
                    for (; slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch; ++slot.n_past)
                    {
                        if (slot.ga_n != 1)
                        {
                            while (slot_npast >= ga_i + ga_w)
                            {
                                const int bd = (ga_w / ga_n) * (ga_n - 1);
                                slot_npast -= bd;
                                ga_i += ga_w / ga_n;
                            }
                        }

                        llama_batch_add(batch, prompt_tokens[slot.n_past], slot_npast, {slot.id + 1}, false);

                        if (slot.params.cache_prompt)
                        {
                            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot_npast++;
                    }

                    /*LOG_VERBOSE("prompt processing progress", {
                                                                  {"id_slot", slot.id},
                                                                  {"n_past", slot.n_past},
                                                                  {"n_ctx", n_ctx},
                                                                  {"n_tokens", batch.n_tokens},
                                                                  {"progress", (float)slot.n_prompt_tokens_processed / slot.n_prompt_tokens},
                                                              });*/

                    // entire prompt has been processed - start decoding new tokens
                    if (slot.n_past == slot.n_prompt_tokens)
                    {
                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch = batch.n_tokens - 1;

                        /*LOG_VERBOSE("prompt done", {
                                                       {"id_slot", slot.id},
                                                       {"n_past", slot.n_past},
                                                       {"n_ctx", n_ctx},
                                                       {"n_tokens", batch.n_tokens},
                                                   });*/
                    }
                }

                if (batch.n_tokens >= n_batch)
                {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0)
        {
            // LOG_VERBOSE("no tokens to decode", {});
            return false;
        }

        /*LOG_VERBOSE("decoding batch", {
                                          {"n_tokens", batch.n_tokens},
                                      });*/

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch)
        {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            for (auto &slot : slots)
            {
                if (slot.ga_n != 1)
                {
                    // context extension via Self-Extend
                    // TODO: simplify and/or abstract this
                    while (slot.n_past_se >= slot.ga_i + slot.ga_w)
                    {
                        const int ib = (slot.ga_n * slot.ga_i) / slot.ga_w;
                        const int bd = (slot.ga_w / slot.ga_n) * (slot.ga_n - 1);
                        const int dd = (slot.ga_w / slot.ga_n) - ib * bd - slot.ga_w;

                        /* LOG_TEE("\n");
                         LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i, slot.n_past_se, ib * bd, slot.ga_i + ib * bd, slot.n_past_se + ib * bd);
                         LOG_TEE("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n, (slot.ga_i + ib * bd) / slot.ga_n, (slot.ga_i + ib * bd + slot.ga_w) / slot.ga_n);
                         LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd, slot.ga_i + ib * bd + slot.ga_w + dd, slot.n_past_se + ib * bd + dd);*/

                        llama_kv_cache_seq_add(ctx, slot.id + 1, slot.ga_i, slot.n_past_se, ib * bd);
                        llama_kv_cache_seq_div(ctx, slot.id + 1, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);
                        llama_kv_cache_seq_add(ctx, slot.id + 1, slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);

                        slot.n_past_se -= bd;

                        slot.ga_i += slot.ga_w / slot.ga_n;

                        // LOG_TEE("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", slot.n_past_se + bd, slot.n_past_se, slot.ga_i);
                    }

                    slot.n_past_se += n_tokens;
                }
            }

            llama_batch batch_view = {
                n_tokens,
                batch.token + i,
                nullptr,
                batch.pos + i,
                batch.n_seq_id + i,
                batch.seq_id + i,
                batch.logits + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);

            if (ret != 0)
            {
                if (n_batch == 1 || ret < 0)
                {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    /*LOG_ERROR("failed to decode the batch: KV cache is full - try increasing it via the context size", {
                                                                                                                            {"i", i},
                                                                                                                            {"n_batch", ret},
                                                                                                                            {"ret", ret},
                                                                                                                        });*/
                    for (auto &slot : slots)
                    {
                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;
                        slot.release();
                        // LOG_VERBOSE("Input prompt is too big compared to KV size. Please try increasing KV size.", {});
                    }
                    break; // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                /*LOG_WARNING("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation", {
                                                                                                                                                                                  {"i", i},
                                                                                                                                                                                  {"n_batch", n_batch},
                                                                                                                                                                                  {"ret", ret},
                                                                                                                                                                              });*/

                continue; // continue loop of n_batch
            }

            for (auto &slot : slots)
            {
                if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens))
                {
                    continue; // continue loop of slots
                }

                completion_token_output result;
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, slot.i_batch - i);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                slot.n_decoded += 1;
                if (slot.n_decoded == 1)
                {
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                }

                llama_token_data_array cur_p = {slot.ctx_sampling->cur.data(), slot.ctx_sampling->cur.size(), false};
                result.tok = id;

                const size_t n_probs = std::min(cur_p.size, (size_t)slot.sparams.n_probs);
                if (n_probs > 0)
                {
                    const size_t n_valid = slot.ctx_sampling->n_valid;

                    // Make sure at least n_probs top tokens are at the front of the vector:
                    if (slot.sparams.temp == 0.0f && n_probs > n_valid)
                    {
                        llama_sample_top_k(ctx, &cur_p, n_probs, 0);
                    }

                    if (slot.sparams.temp == 0.0f)
                    {
                        // With greedy sampling the probabilities have possibly not been calculated.
                        for (size_t i = 0; i < n_probs; ++i)
                        {
                            result.probs.push_back({cur_p.data[i].id,
                                                    i == 0 ? 1.0f : 0.0f});
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < n_probs; ++i)
                        {
                            result.probs.push_back({
                                cur_p.data[i].id,
                                i >= n_valid ? 0.0f : cur_p.data[i].p // Tokens filtered out due to e.g. top_k have 0 probability.
                            });
                        }
                    }
                }

                if (!process_token(result, slot))
                {
                    slot.release();
                    send_final_response(&slot);
                }

                slot.i_batch = -1;
            }
        }

        // LOG_VERBOSE("run slots completed", {});

        return true;
    }
};

// struct that contains llama context and inference
server_context ctx_server;

LIB_API int LLAMA_CheckLoops()
{
    return loops;
}

LIB_API int LLAMA_Terminate()
{
    return ctx_server.queue_tasks.running = false;
}

LIB_API char *LLAMA_GetData(uint32_t slotIndex, uint32_t &dataSize)
{
    auto slot = &ctx_server.slots[slotIndex];
    std::stringstream ss;
    int end = slot->tkns_pos;
    while (slot->tkns_start != end)
    {
        ss << slot->tkns[slot->tkns_start];
        slot->tkns_start = (slot->tkns_start + 1) % slot->tkns_length;
    }

    auto str = ss.str();
    auto strdata = str.data();

    dataSize = str.length();

    char *memcpydata;
    memcpydata = (char *)malloc(dataSize);
    memcpy(memcpydata, (const char *)strdata, dataSize);

    return memcpydata;
}

LIB_API void LLAMA_DiscardData(char *data)
{
    free(data);
}

LIB_API bool LLAMA_IsActive(uint32_t slotIndex)
{
    server_slot *slot = &ctx_server.slots[slotIndex];

    return slot->active || slot->tkns_start != slot->tkns_pos;
}

LIB_API bool LLAMA_Prompt(uint32_t slotIndex, const char *text)
{
    if (slotIndex >= ctx_server.slots.size())
        return false;

    try
    {
        server_slot *slot = &ctx_server.slots[slotIndex];

        if (slot->active)
            return false;

        auto data = json::parse(text);

        server_task task;
        task.id = 0;
        task.id_target = 0;
        task.data = std::move(data);
        task.infill = false;
        task.embedding = false;
        task.type = SERVER_TASK_TYPE_COMPLETION;

        slot->reset();

        slot->id_task = task.id;
        slot->infill = task.infill;
        slot->embedding = task.embedding;

        ctx_server.launch_slot_with_task(*slot, task);
    }
    catch (...)
    {
        return false;
    }

    return true;
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal)
{
    if (is_terminating.test_and_set())
    {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

LIB_API int main(int argc, char **argv)
{

#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params))
    {
        gpt_params_print_usage(argc, argv, params);
        return 420;
    }

    // TODO: not great to use extern vars
    server_log_json = params.log_json;
    server_verbose = params.verbosity > 0;

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    if (!ctx_server.load_model(params))
    {
        return 1337;
    }
    else
    {
        ctx_server.init();
    }

    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));

    shutdown_handler = [&](int)
    {
        ctx_server.queue_tasks.terminate();
    };

    ctx_server.queue_tasks.start_loop();

    llama_backend_free();

    return 0;
}