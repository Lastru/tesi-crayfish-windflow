#ifndef INFERENCE_BACKEND_HPP
#define INFERENCE_BACKEND_HPP

#include <vector>
#include <string>


// Interfaccia astratta per il backend di inferenza
// Ogni nuova tecnologia deve implementare questa classe
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // Metodo principale: prende dati numerici, restituisce una predizione
    virtual std::vector<float> predict(const std::vector<float>& input) = 0;

    // debug
    virtual std::string backend_name() const = 0;
};

#endif
