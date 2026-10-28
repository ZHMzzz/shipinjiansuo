// A small adapter around the unmodified, pinned llama.cpp / libmtmd libraries.
// Preserves the production Qwen template, timestamps, paired video frames and EOS.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#import <Metal/Metal.h>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

struct MetalMonitor {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> peak{0};
    std::thread sampler;
    MetalMonitor() : sampler([this] {
        while (!stop.load()) {
            uint64_t current = [device currentAllocatedSize];
            if (current > peak.load()) peak.store(current);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }) {}
    uint64_t current() { return [device currentAllocatedSize]; }
    ~MetalMonitor() { stop.store(true); sampler.join(); [device release]; }
};

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    MetalMonitor metal;
    auto start = clock_type::now();
    ggml_backend_load_all_from_path(argv[3]);
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        std::cerr << "Metal GPU backend unavailable\n";
        return 6;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto vp = mtmd_context_params_default();
    vp.use_gpu = true;
    vp.warmup = false;
    vp.n_threads = 4;
    vp.image_min_tokens = 4;
    vp.image_max_tokens = 128;
    vp.batch_max_tokens = 2048;
    auto * vision = mtmd_init_from_file(argv[2], model, vp);
    if (!vision) return 5;
    std::cout << json({{"ready", true}, {"metal_allocated_bytes", metal.current()},
        {"metal_peak_bytes", metal.peak.load()},
        {"load_seconds", std::chrono::duration<double>(clock_type::now()-start).count()}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto req = json::parse(line);
            if (req.value("quit", false)) break;
            metal.peak.store(metal.current()); // per-request peak, parent aggregates by phase
            auto t = clock_type::now();
            llama_memory_clear(llama_get_memory(ctx), true);
            std::vector<mtmd_bitmap *> bitmaps;
            std::vector<std::string> texts;
            std::vector<mtmd_input_text> text_parts;
            std::vector<mtmd_input_part> parts;
            // Reserve to keep all string and struct pointers stable.
            texts.reserve(32); text_parts.reserve(32); parts.reserve(64);
            auto add_text = [&](const std::string & value) {
                texts.push_back(value);
                auto & s = texts.back();
                text_parts.push_back({s.c_str(), s.size(), false, true});
                parts.push_back({&text_parts.back(), nullptr});
            };
            if (req.contains("rgb_path")) {
                int w = req.at("width"), h = req.at("height");
                std::ifstream f(req.at("rgb_path").get<std::string>(), std::ios::binary);
                if (!f) throw std::runtime_error("Cannot open RGB frames");
                std::vector<unsigned char> rgb(size_t(w)*h*3);
                add_text(req.at("prefix"));
                for (int i=0; i<16; ++i) {
                    if (i % 2 == 0) add_text(req.at("timestamps").at(i/2));
                    f.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
                    if (f.gcount() != (std::streamsize)rgb.size()) throw std::runtime_error("Truncated RGB frames");
                    auto * bitmap = mtmd_bitmap_init(w, h, rgb.data());
                    mtmd_bitmap_set_mergeable(bitmap, true);
                    bitmaps.push_back(bitmap);
                    parts.push_back({nullptr, bitmap});
                }
                add_text(req.at("suffix"));
            } else {
                add_text(req.at("prompt"));
            }
            std::vector<const mtmd_input_part *> pointers;
            for (auto & p : parts) pointers.push_back(&p);
            auto * chunks = mtmd_input_chunks_init();
            int rc = mtmd_tokenize_from_parts(vision, chunks, pointers.data(), pointers.size(), true);
            if (rc) throw std::runtime_error("mtmd tokenize failed: " + std::to_string(rc));
            json input_info = {{"tokens", 0}, {"visual_chunk_tokens", json::array()}, {"text_token_ids", json::array()}};
            for (size_t i=0; i<mtmd_input_chunks_size(chunks); ++i) {
                auto * c = mtmd_input_chunks_get(chunks, i);
                auto n = mtmd_input_chunk_get_n_tokens(c);
                input_info["tokens"] = input_info["tokens"].get<int>() + n;
                if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t count;
                    auto * ids = mtmd_input_chunk_get_tokens_text(c, &count);
                    for (size_t j=0;j<count;++j) input_info["text_token_ids"].push_back(ids[j]);
                } else {
                    input_info["visual_chunk_tokens"].push_back(n);
                }
            }
            llama_pos end_pos = 0;
            rc = mtmd_helper_eval_chunks(vision, ctx, chunks, 0, 0, 2048, true, &end_pos);
            if (rc) throw std::runtime_error("mtmd eval failed: " + std::to_string(rc));
            llama_synchronize(ctx);
            const float * emb = llama_get_embeddings_seq(ctx, 0);
            if (!emb) throw std::runtime_error("No pooled embedding");
            int dim = llama_model_n_embd(model);
            double norm = 0;
            for (int i=0;i<dim;++i) norm += double(emb[i])*emb[i];
            norm = std::sqrt(norm);
            if (!std::isfinite(norm) || norm < 1e-8) throw std::runtime_error("Invalid embedding norm");
            std::vector<float> vec(dim);
            for (int i=0;i<dim;++i) vec[i] = emb[i] / norm;
            double seconds = std::chrono::duration<double>(clock_type::now()-t).count();
            mtmd_input_chunks_free(chunks);
            for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
            std::cout << json({{"embedding", vec}, {"seconds", seconds}, {"input", input_info},
                {"metal_allocated_bytes", metal.current()}, {"metal_peak_bytes", metal.peak.load()}}).dump() << std::endl;
        } catch (const std::exception & e) {
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
            break; // fail fast; parent treats this as a failed experiment
        }
    }
    mtmd_free(vision);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
