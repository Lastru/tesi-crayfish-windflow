#ifndef TF_SERVING_BACKEND_HPP
#define TF_SERVING_BACKEND_HPP

#include "InferenceBackend.hpp"
#include "NetworkUtils.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <unordered_map>
#include <iomanip> 

using json = nlohmann::json;

class TfServingBackend : public InferenceBackend {
public:
    static thread_local int t_sockfd;
    static thread_local std::string t_body_buffer;

    TfServingBackend(const std::string& model_config, const std::string& model_name, std::unordered_map<std::string, std::string> (*load_props)(const std::string&)) {
        auto props = load_props(model_config);
        std::string tfserving_path = props["model.path.tf-serving"];
        model_name_ = props["model.name"].empty() ? model_name : props["model.name"];
        input_name_ = props["input.name"].empty() ? "input_1" : props["input.name"];
        host_ = tfserving_path; port_ = 8501;
        size_t pos = tfserving_path.find(':');
        if (pos != std::string::npos) {
            host_ = tfserving_path.substr(0, pos);
            try { port_ = std::stoi(tfserving_path.substr(pos + 1)); } catch (...) {}
        }
        if (port_ == 8500) port_ = 8501;
    }

    std::vector<float> predict(const std::vector<float>& input) override {
        t_body_buffer.clear();
        if (t_body_buffer.capacity() < 16384) t_body_buffer.reserve(16384);

        std::ostringstream ss;
        ss << std::setprecision(17);
        
        // Logica ResNet50 (Nidificata [224][224][3])
        if (model_name_ == "resnet50" && input.size() == 3u * 224u * 224u) {
            ss << "{\"instances\":[{\"" << input_name_ << "\":[";
            for (int h = 0; h < 224; ++h) {
                ss << (h > 0 ? ",[" : "[");
                for (int w = 0; w < 224; ++w) {
                    ss << (w > 0 ? ",[" : "[");
                    for (int c = 0; c < 3; ++c) {
                        size_t idx = (size_t)c * 224u * 224u + (size_t)h * 224u + (size_t)w;
                        ss << input[idx] << (c < 2 ? "," : "");
                    }
                    ss << "]";
                }
                ss << "]";
            }
            ss << "]}]";
        } 
        
        // Logica FFNN
        else {
            ss << "{\"instances\":[{\"" << input_name_ << "\":[";
            for (size_t i = 0; i < input.size(); ++i) {
                ss << input[i];
                if (i < input.size() - 1) ss << ",";
            }
            ss << "]}]";
        }

        // Chiusura comune
        ss << ",\"signature_name\":\"serving_default\"}";
        t_body_buffer = ss.str();

        std::string path = "/v1/models/" + model_name_ + ":predict";
        std::string resp = simple_http_post_persistent(t_sockfd, host_, port_, path, t_body_buffer, "application/json");

        if (resp.empty()) return {};

        try {
            json j_resp = json::parse(resp);
            if (j_resp.contains("predictions") && !j_resp["predictions"].empty()) {
                return j_resp["predictions"][0].get<std::vector<float>>();
            }
        } catch (...) {}
        return {};
    }

    std::string backend_name() const override { return "TF_Serving_Optimized_Hybrid"; }

private:
    std::string host_, model_name_, input_name_;
    int port_;
};

thread_local int TfServingBackend::t_sockfd = -1;
thread_local std::string TfServingBackend::t_body_buffer = "";

#endif
