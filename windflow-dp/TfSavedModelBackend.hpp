#ifndef TF_SAVEDMODEL_BACKEND_HPP
#define TF_SAVEDMODEL_BACKEND_HPP

#include "InferenceBackend.hpp"
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <unordered_map>


// Backend Embedded TF-SavedModel
class TfSavedModelBackend : public InferenceBackend {
	
public:
    TfSavedModelBackend(const std::string& model_config, const std::string& model_name, std::unordered_map<std::string, std::string> (*load_props)(const std::string&)) {
        std::cerr << "[WF-DP] Initializing SavedModel embedded backend...\n";
        
        auto props = load_props(model_config);
        std::string savedmodel_path = props["model.path.tf-savedmodel"];
        
        if (savedmodel_path.empty()) {
            std::cerr << "[WF-DP] ERROR: 'model.path.tf-savedmodel' not found in config.\n";
            is_valid_ = false;
			return;
        }
        

        if (model_name == "ffnn" || model_name == "FFNN") {
			input_node_dims_ = {1, 784};
        	savedmodel_input_name_ = "serving_default_input_1";
        	savedmodel_output_name_ = "StatefulPartitionedCall";
		}
        else if (model_name == "resnet50" || model_name == "RESNET50") { 
			input_node_dims_ = {1, 3, 224, 224}; 
        	savedmodel_input_name_ = "serving_default_input";
        	savedmodel_output_name_ = "StatefulPartitionedCall";
		}
        else { input_node_dims_ = {1, 1, 1}; }
        

        tf_status_ = TF_NewStatus();
        tf_graph_  = TF_NewGraph();
        tf_session_opts_ = TF_NewSessionOptions();
        const char* tags[] = { "serve" };
        
		tf_session_ = TF_LoadSessionFromSavedModel(tf_session_opts_, nullptr, savedmodel_path.c_str(), tags, 1, tf_graph_, nullptr, tf_status_);

        if (TF_GetCode(tf_status_) != TF_OK) {
            std::cerr << "[WF-DP] ERROR loading SavedModel: " << TF_Message(tf_status_) << "\n";
            is_valid_ = false;
			return;
        } else {
            std::cerr << "[WF-DP] SavedModel loaded successfully.\n";
        }
    }

    ~TfSavedModelBackend() {
        if (tf_session_) { TF_CloseSession(tf_session_, tf_status_); TF_DeleteSession(tf_session_, tf_status_); }
        if (tf_session_opts_) { TF_DeleteSessionOptions(tf_session_opts_); }
        if (tf_graph_) { TF_DeleteGraph(tf_graph_); }
        if (tf_status_) { TF_DeleteStatus(tf_status_); }
    }

    std::vector<float> predict(const std::vector<float>& input) override {
        if (!is_valid_ || !tf_session_ || input.empty()) return {};

        std::vector<int64_t> dims = input_node_dims_;
        size_t bytes_tensor = input.size() * sizeof(float); 

        TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, dims.data(), dims.size(), bytes_tensor);
        std::memcpy(TF_TensorData(input_tensor), input.data(), bytes_tensor);

        TF_Operation* in_op = TF_GraphOperationByName(tf_graph_, savedmodel_input_name_.c_str());
        TF_Operation* out_op = TF_GraphOperationByName(tf_graph_, savedmodel_output_name_.c_str());
        if (!in_op || !out_op) { TF_DeleteTensor(input_tensor); return {}; }

        TF_Output tf_input = {in_op, 0};
        TF_Output tf_output = {out_op, 0};
        TF_Tensor* output_tensor = nullptr;

        TF_SessionRun(tf_session_, nullptr, &tf_input, &input_tensor, 1, &tf_output, &output_tensor, 1, nullptr, 0, nullptr, tf_status_);
        TF_DeleteTensor(input_tensor);

        if (TF_GetCode(tf_status_) != TF_OK || !output_tensor) {
            std::cerr << "[WF-DP] TF Run Error: " << TF_Message(tf_status_) << "\n";
            if (output_tensor) TF_DeleteTensor(output_tensor);
            return {};
        }

        float* out_data = static_cast<float*>(TF_TensorData(output_tensor));
        size_t out_elems = TF_TensorByteSize(output_tensor) / sizeof(float);
        std::vector<float> res(out_data, out_data + out_elems);
        TF_DeleteTensor(output_tensor);
        return res;
    }

    std::string backend_name() const override { return "Embedded_SavedModel"; }

private:
    TF_Graph* tf_graph_ = nullptr;
    TF_Session* tf_session_ = nullptr;
    TF_Status* tf_status_ = nullptr;
    TF_SessionOptions* tf_session_opts_ = nullptr;
    
    std::string savedmodel_input_name_, savedmodel_output_name_;
    std::vector<int64_t> input_node_dims_;
    
    bool is_valid_ = true;
};

#endif
