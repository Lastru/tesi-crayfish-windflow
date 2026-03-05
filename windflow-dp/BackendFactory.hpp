#ifndef BACKEND_FACTORY_HPP
#define BACKEND_FACTORY_HPP

#include <memory>
#include <unordered_map>
#include "CmdOptions.hpp"

#include "InferenceBackend.hpp"
#include "OnnxBackend.hpp"
#include "TfSavedModelBackend.hpp"
#include "TfServingBackend.hpp"
#include "TorchServeBackend.hpp"
#include "DummyBackend.hpp"

std::unordered_map<std::string, std::string> load_properties(const std::string& path);


// Creatore Dinamico dei Motori di Inferenza
class BackendFactory {
public:
    static std::unique_ptr<InferenceBackend> create(const CmdOptions& opts) {
    	
        if (opts.model_format == "onnx") {
        	
            return std::make_unique<OnnxBackend>(opts.model_config, opts.model_name, load_properties);
            
        } else if (opts.model_format == "tf-savedmodel") {
        	
            return std::make_unique<TfSavedModelBackend>(opts.model_config, opts.model_name, load_properties);
            
        } else if (opts.model_format == "tf-serving") {
        	
            return std::make_unique<TfServingBackend>(opts.model_config, opts.model_name, load_properties);
            
        } else if (opts.model_format == "torchserve") {
        	
            return std::make_unique<TorchServeBackend>(opts.model_config, opts.model_name, load_properties);
            
        }
        
        return std::make_unique<DummyBackend>(opts.model_name);
    }
};

#endif
