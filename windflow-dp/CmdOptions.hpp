#ifndef CMD_OPTIONS_HPP
#define CMD_OPTIONS_HPP

#include <string>


// Struttura Dati per la Configurazione e i Parametri di Esecuzione
struct CmdOptions {
    std::string model_format;      // onnx, tf-savedmodel, tf-serving, torchserve
    std::string model_name;        // ffnn, resnet50
    std::string model_config;
    std::string global_config;
    std::string experiment_config;
    std::string kafka_input_topic;
    std::string kafka_output_topic;
    char task_parallel = 't'; 	  // data_parallel, task_parallel
    bool is_embedded = true; 
};

#endif
