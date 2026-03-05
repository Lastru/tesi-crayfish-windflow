#ifndef TORCHSERVE_BACKEND_HPP
#define TORCHSERVE_BACKEND_HPP

#include "InferenceBackend.hpp"
#include "NetworkUtils.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

class TorchServeBackend : public InferenceBackend {
public:
    static thread_local int t_sockfd;

     TorchServeBackend(const std::string& model_config, const std::string& model_name, std::unordered_map<std::string, std::string> (*load_props)(const std::string&)) {
        auto props = load_props(model_config);
        std::string ts_path = props["model.path.torchserve"]; 
        if (ts_path.empty()) ts_path = "localhost:8080";
        model_name_ = props["model.name"].empty() ? model_name : props["model.name"];
        host_ = ts_path; port_ = 8080; 
        size_t pos = ts_path.find(':');
        if (pos != std::string::npos) {
            host_ = ts_path.substr(0, pos);
            try { port_ = std::stoi(ts_path.substr(pos + 1)); } catch (...) {}
        }
        if (port_ == 7070) port_ = 8080;
    }

    std::vector<float> predict(const std::vector<float>& input) override {
        std::string path = "/predictions/" + model_name_;
        std::string payload;
        std::string content_type = "application/octet-stream";

        if (model_name_ == "resnet50" && input.size() == 150528) {
            std::ostringstream ss;
            ss.str().reserve(1300000); 
            ss << std::setprecision(17);
            
            // Logica ResNet50
            ss << "[[[["; 
            for (int c = 0; c < 3; ++c) { 
                if (c > 0) ss << "],[";
                for (int h = 0; h < 224; ++h) {
                    if (h > 0) ss << "],[";
                    for (int w = 0; w < 224; ++w) {
                        if (w > 0) ss << ",";
                        // Mappatura: WindFlow (H,W,C) -> Torch (C,H,W)
                        size_t idx_nhwc = (static_cast<size_t>(h) * 224u + w) * 3u + c;
                        ss << input[idx_nhwc];
                    }
                }
            }
            ss << "]]]]";
            payload = ss.str();
        } 
        else {
            // Logica FFNN
            payload = json::array({input}).dump();
        }

        std::string response_str = simple_http_post_persistent(t_sockfd, host_, port_, path, payload, content_type);

        if (response_str.empty()) return {};

        try {
            if (response_str[0] == '{') {
                json j_resp = json::parse(response_str);
                if (j_resp.contains("prediction")) {
                    auto& p = j_resp["prediction"];
                    auto& data = (p.is_array() && !p.empty() && p[0].is_array()) ? p[0] : p;
                    return data.get<std::vector<float>>();
                }
            } else if (response_str[0] == '[') {
                json j_resp = json::parse(response_str);
                if (!j_resp.empty()) {
                    auto& data = j_resp[0].is_array() ? j_resp[0] : j_resp;
                    return data.get<std::vector<float>>();
                }
            } else if (response_str.find("tensor(") != std::string::npos) {
                size_t start = response_str.find('[');
                size_t end = response_str.rfind(']');
                if (start != std::string::npos && end != std::string::npos) {
                    return parse_datapoint_string(response_str.substr(start + 1, end - start - 1));
                }
            }
        } catch (...) {}
        return {};
    }

    std::string backend_name() const override { return "TorchServe_NCHW_Final"; }

private:
    std::string host_, model_name_;
    int port_;
    std::vector<float> parse_datapoint_string(const std::string& s) {
        std::vector<float> result; result.reserve(1024);
        std::stringstream ss(s); std::string item;
        while (std::getline(ss, item, ',')) {
            size_t start = item.find_first_of("-0123456789.");
            size_t end = item.find_last_of("0123456789");
            if (start == std::string::npos) continue;
            try { result.push_back(std::stof(item.substr(start, end - start + 1))); } catch (...) {}
        }
        return result;
    }
};

thread_local int TorchServeBackend::t_sockfd = -1;

#endif
