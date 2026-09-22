#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(parse_json_body(req), limits);
        validate_openai_model(request.model, public_model_id_);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        const ninfer::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     observation, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress,
             timings_per_token](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };
                try {
                    render_and_write(transport, [&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const ninfer::PromptProgress& progress) {
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    RequestJson body;
    GenerationRequest request;
    std::string model;
    bool stream = false;
    try {
        body               = parse_json_body(req);
        model              = body.value("model", public_model_id_);
        validate_openai_model(model, public_model_id_);
        request.raw_output = true;
        request.max_tokens = body.value("max_tokens", options_.default_max_tokens);
        stream             = body.value("stream", false);

        if (body.contains("prompt") && !body["prompt"].is_null()) {
            const auto& p = body["prompt"];
            if (p.is_array() && !p.empty() && p[0].is_number_integer()) {
                request.raw_tokens = p.get<std::vector<ninfer::TokenId>>();
            } else if (p.is_string()) {
                request.raw_tokens = service_->tokenize_text(p.get<std::string>());
            }
        } else {
            request.raw_tokens = std::vector<ninfer::TokenId>{};
        }

        if (body.contains("temperature") && !body["temperature"].is_null()) {
            request.sampling.temperature = body["temperature"].get<double>();
        }
        if (body.contains("top_p") && !body["top_p"].is_null()) {
            request.sampling.top_p = body["top_p"].get<double>();
        }
        if (body.contains("stop") && !body["stop"].is_null()) {
            if (body["stop"].is_string()) {
                request.stop_strings.push_back(body["stop"].get<std::string>());
            } else if (body["stop"].is_array()) {
                for (const auto& s : body["stop"]) {
                    if (s.is_string()) { request.stop_strings.push_back(s.get<std::string>()); }
                }
            }
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& e) {
        ApiError err{.status = 400, .message = e.what()};
        write_openai_error(res, err);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = model,
                                      .stream                 = stream,
                                      .output_tokens_explicit = body.contains("max_tokens")};
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(request,
                                     stream ? GenerationConsumerMode::Streaming
                                            : GenerationConsumerMode::Aggregate,
                                     {}, [&req] { return client_disconnected(req); });
    } catch (const ApiException& ex) {
        write_openai_error(res, ex.error());
        return;
    } catch (const std::exception& ex) {
        ApiError err{.status = 500, .message = ex.what()};
        write_openai_error(res, err);
        return;
    }

    auto lifecycle = begin_request(
        make_request_log_context(req_id, "openai_completions", request, metadata, prepared));

    if (!stream) {
        GenerationOutcome outcome;
        try {
            outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const std::exception& ex) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, ex.what()));
            ApiError err{.status = 500, .message = ex.what()};
            write_openai_error(res, err);
            return;
        }
        lifecycle->done(outcome);

        const char* finish =
            outcome.finish_reason == ninfer::FinishReason::OutputLimit ? "length" : "stop";
        nlohmann::json resp = {
            {"id", new_openai_completion_id()},
            {"object", "text_completion"},
            {"created", unix_time_now()},
            {"model", model},
            {"choices", nlohmann::json::array({{{"text", outcome.text},
                                                {"index", 0},
                                                {"logprobs", nullptr},
                                                {"finish_reason", finish},
                                                {"token_ids", outcome.generated_token_ids}}})},
            {"usage",
             {{"prompt_tokens", outcome.prompt_tokens},
              {"completion_tokens", outcome.completion_tokens},
              {"total_tokens", outcome.prompt_tokens + outcome.completion_tokens}}}};
        set_owned_json_content(res, resp.dump(), prepared.lifetime);
        return;
    }

    try {
        auto stream_state          = std::make_shared<HttpGenerationStream>(std::move(prepared));
        const std::string cmpl_id  = new_openai_completion_id();
        const std::int64_t created = unix_time_now();
        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream_state, lifecycle, cmpl_id, model,
             created](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream_state->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream_state->cancelled);
                StreamSink consumer{
                    .on_content =
                        [&](const std::string& delta) {
                            nlohmann::json chunk = {
                                {"id", cmpl_id},
                                {"object", "text_completion"},
                                {"created", created},
                                {"model", model},
                                {"choices", nlohmann::json::array({{{"text", delta},
                                                                    {"index", 0},
                                                                    {"logprobs", nullptr},
                                                                    {"finish_reason", nullptr}}})}};
                            transport.write("data: " + chunk.dump() + "\n\n");
                        },
                    .is_cancelled =
                        [&] { return stream_state->cancelled.load(std::memory_order_acquire); }};

                GenerationOutcome outcome;
                try {
                    outcome =
                        service_->run(stream_state->prepared, &consumer, consumer.is_cancelled);
                } catch (const std::exception& ex) {
                    lifecycle->failure(
                        make_internal_request_failure(RequestFailurePhase::Generation, ex.what()));
                    sink.done();
                    return false;
                }
                lifecycle->done(outcome);

                const char* finish =
                    outcome.finish_reason == ninfer::FinishReason::OutputLimit ? "length" : "stop";
                nlohmann::json final_chunk = {
                    {"id", cmpl_id},
                    {"object", "text_completion"},
                    {"created", created},
                    {"model", model},
                    {"choices", nlohmann::json::array({{{"text", ""},
                                                        {"index", 0},
                                                        {"logprobs", nullptr},
                                                        {"finish_reason", finish}}})}};
                transport.write("data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n");
                sink.done();
                return true;
            },
            [stream_state, lifecycle](bool successful) {
                stream_state->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream_state->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& ex) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, ex.what()));
        ApiError err{.status = 500, .message = ex.what()};
        write_openai_error(res, err);
    }
}

} // namespace ninfer::serve
