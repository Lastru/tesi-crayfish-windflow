#ifndef DUMMY_BACKEND_HPP
#define DUMMY_BACKEND_HPP

#include "InferenceBackend.hpp"
#include <vector>
#include <string>


// Simulatore di Inferenza per Test e Fallback
class DummyBackend : public InferenceBackend {
	
public:
    DummyBackend(const std::string& model_name) {
        if (model_name == "ffnn" || model_name == "FFNN") kind = 1;
        else if (model_name == "resnet50" || model_name == "RESNET50") kind = 2;
    }

    std::vector<float> predict(const std::vector<float>& input) override {
        std::vector<float> p;
        if (kind == 1) p.assign(10, 0.0f);
        else if (kind == 2) p.assign(1000, 0.0f);
        else p.assign(1, 0.0f);
        if (!p.empty()) p[0] = 1.0f;
        return p;
    }
    
    std::string backend_name() const override { return "Dummy_Fallback"; }

private:
    int kind = 0; // 1: FFNN, 2: ResNet
};

#endif
