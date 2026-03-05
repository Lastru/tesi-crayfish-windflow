#ifndef ONNX_BACKEND_HPP
#define ONNX_BACKEND_HPP

#include "InferenceBackend.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <memory>
#include <vector>


// Backend Embedded ONNX
class OnnxBackend : public InferenceBackend {
	
public:
    OnnxBackend(const std::string& model_config, const std::string& model_name, std::unordered_map<std::string, std::string> (*load_props)(const std::string&)) {
        std::cerr << "[WF-DP] Initializing ONNX embedded backend...\n";
        
        auto props = load_props(model_config);
        std::string onnx_model_path = props["model.path.onnx"];
        
        if (onnx_model_path.empty()) {
            std::cerr << "[WF-DP] ERROR: 'model.path.onnx' not found in config.\n";
            is_valid_ = false;
			return;
        }
        
        
        if (model_name == "ffnn" || model_name == "FFNN") {
        	onnx_model_path += "model-ffnn.onnx";
            input_node_dims_ = {1, 1, 784};
	        input_node_names_.push_back("input");
	        output_node_names_.push_back("output");
        } 
		else if (model_name == "resnet50" || model_name == "RESNET50") {
			onnx_model_path += "model-resnet50.onnx";
            input_node_dims_ = {1, 3, 224, 224};
	        input_node_names_.push_back("input");
	        output_node_names_.push_back("output");
        }
        else { input_node_dims_ = {1, 1, 1}; }
        
        
        try {
            env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "wf-dp");
            session_options_ = std::make_shared<Ort::SessionOptions>();
            session_options_->SetIntraOpNumThreads(1);
            session_options_->SetGraphOptimizationLevel(ORT_ENABLE_ALL);

            session_ = std::make_shared<Ort::Session>(*env_, onnx_model_path.c_str(), *session_options_);
            std::cerr << "[WF-DP] ONNX session created. Path: " << onnx_model_path << "\n";
        } catch (const Ort::Exception& e) {
            std::cerr << "[WF-DP] ERROR creating ONNX session: " << e.what() << "\n";
            is_valid_ = false; 
			return;
        }

    }

    std::vector<float> predict(const std::vector<float>& input) override {
        if (!is_valid_ || !session_ || input.empty()) return {};
        
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(input.data()), input.size(), input_node_dims_.data(), input_node_dims_.size());

        try {
            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), 1);
            if (output_tensors.empty()) return {};
            
            float* out_data = output_tensors[0].GetTensorMutableData<float>();
            size_t out_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            return std::vector<float>(out_data, out_data + out_size);
        } catch (const std::exception& e) {
            std::cerr << "[WF-DP] ONNX Run error: " << e.what() << "\n";
            return {};
        }
    }

    std::string backend_name() const override { return "Embedded_ONNX"; }

private:
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::SessionOptions> session_options_;
    std::shared_ptr<Ort::Session> session_;
    
    std::vector<const char*> input_node_names_, output_node_names_;
    std::vector<int64_t> input_node_dims_;
    
    bool is_valid_ = true;
};

#endif
