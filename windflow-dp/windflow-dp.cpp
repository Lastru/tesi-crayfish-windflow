#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
#include <unistd.h>

#include "wf/windflow.hpp"
#include "wf/kafka/windflow_kafka.hpp"

#include <nlohmann/json.hpp>
using nlohmann::json;

#include "CmdOptions.hpp"
#include "BackendFactory.hpp"
#include "InferenceBackend.hpp"

static std::atomic<long> g_source_msgs{0};
static std::atomic<long> g_sink_msgs{0};


// Analizzatore della Riga di Comando
CmdOptions parse_args(int argc, char** argv) {
    CmdOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](std::string& target) {
            if (i + 1 < argc) { target = argv[++i]; } 
            else { std::cerr << "[WF-DP] Missing value after " << arg << "\n"; std::exit(1); }
        };

        if (arg == "--model-format") next(opts.model_format);
        else if (arg == "--model-name") next(opts.model_name);
        else if (arg == "--model-config") next(opts.model_config);
        else if (arg == "--global-config") next(opts.global_config);
        else if (arg == "--experiment-config") next(opts.experiment_config);
        else if (arg == "--kafka-input") next(opts.kafka_input_topic);
        else if (arg == "--kafka-output") next(opts.kafka_output_topic);
        else if (arg == "--task-par") {
            std::string tmp; next(tmp);
            if (tmp == "true" || tmp == "TRUE") opts.task_parallel = 't';
            else if (tmp == "false" || tmp == "FALSE") opts.task_parallel = 'd';
        } else if (arg == "--is-embedded") {
            std::string tmp; next(tmp);
            opts.is_embedded = (tmp == "true" || tmp == "TRUE" || tmp == "1");
        } else { std::cerr << "[WF-DP] Unknown argument: " << arg << "\n"; }
    }
    return opts;
}


// Lettore del File di Configurazione
std::unordered_map<std::string, std::string> load_properties(const std::string& path) {
    std::unordered_map<std::string, std::string> props;
    std::ifstream in(path);
    if (!in) { std::cerr << "[WF-DP] Cannot open properties/YAML file: " << path << "\n"; return props; }

    std::string line;
    while (std::getline(in, line)) {
        auto trim = [](std::string& s) {
            const char* ws = " \t\r\n";
            size_t start = s.find_first_not_of(ws);
            size_t end   = s.find_last_not_of(ws);
            if (start == std::string::npos || end == std::string::npos) s.clear();
            else s = s.substr(start, end - start + 1);
        };
        std::replace(line.begin(), line.end(), '\t', ' ');
        trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        size_t pos_equal = line.find('=');
        size_t pos_colon = line.find(':');
        size_t pos = std::string::npos;
        if (pos_equal != std::string::npos && (pos_colon == std::string::npos || pos_equal < pos_colon)) pos = pos_equal;
        else if (pos_colon != std::string::npos) pos = pos_colon;
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        trim(key); 
		trim(value);
        if (!key.empty()) props[key] = value;
    }
    return props;
}


// Alias di Tipo per il Messaggio di Scambio
using tuple_t = std::string;


//Motore di Elaborazione e Predizione
class Inference_Functor {
	
public:
    explicit Inference_Functor(const CmdOptions& opts) {
        backend_ = BackendFactory::create(opts);
    }

    tuple_t operator()(const tuple_t& in_json) {
    	
        // std::cerr << "[WF-DP] Inference_Functor called\n";
        
		static std::atomic<long> n_infer{0};
		long n = n_infer.fetch_add(1, std::memory_order_relaxed) + 1;
		if (n % 1000 == 0) {
		    std::cerr << "[WF-DP] Inference_Functor processed " << n << " messages\n";
		}
		
		try {
            json j = json::parse(in_json);
            std::vector<float> input_vec;
            std::string dp_str;
            if (j.contains("dataPoint") && j["dataPoint"].is_string()) {
                dp_str = j["dataPoint"].get<std::string>();
            }
            input_vec = parse_datapoint_string(dp_str);

            std::vector<float> prediction = backend_->predict(input_vec);
			
			if (prediction.empty()) {
			    std::cerr << "[ERROR-DP] Prediction is EMPTY" << std::endl;
			}
            
            j["prediction"] = json::array({ prediction });
            
            std::ostringstream dp_preview;
			dp_preview << "";
			for (size_t i = 0; i < std::min<size_t>(5, input_vec.size()); ++i) {
			    dp_preview << input_vec[i];
			    if (i + 1 < std::min<size_t>(5, input_vec.size())) dp_preview << ",";
			}
			dp_preview << ""; 
			
			j["dataPoint"] = dp_preview.str();
			
			j["dataPointSz2"] = 5;
            
            // std::cerr << "[WF-DP] Output JSON: " << j.dump() << "\n"; // Output Dati e Prediction
            return j.dump();
            
        } catch (const std::exception& ex) {
            std::cerr << "[WF-DP] Inference_Functor JSON error: " << ex.what() << "\n";
            return in_json;
        }
    }

private:
    std::unique_ptr<InferenceBackend> backend_;
    
    static std::vector<float> parse_datapoint_string(const std::string& s) {
        std::vector<float> result; result.reserve(1024);
        std::stringstream ss(s); std::string item;
        while (std::getline(ss, item, ',')) {
            size_t start = item.find_first_not_of(" \t\r\n");
            size_t end = item.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) try { result.push_back(std::stof(item.substr(start, end - start + 1))); } catch (...) {}
        }
        return result;
    }
};


// Gestore del Flusso di Ingresso Dati da Kafka
class KafkaSource_Functor {

public:
    bool operator()(std::optional<std::reference_wrapper<RdKafka::Message>> msg, wf::Source_Shipper<tuple_t> &shipper) {
		if (!msg) { 
			// std::cerr << "[WF-DP][SRC] msg=<none> (idle/timeout)\n";
			return true; 
		} // idle
        
		RdKafka::Message &m = msg->get();
        
        if (m.err() == RdKafka::ERR_NO_ERROR) {
            const void *payload = m.payload();
            size_t len = static_cast<size_t>(m.len());
            if (payload && len > 0) {
                tuple_t out(static_cast<const char*>(payload), len);
                g_source_msgs++;
                if (g_source_msgs % 100 == 0) std::cerr << "[WF-DP] Source processed " << g_source_msgs.load() << " messages\n";
                shipper.push(std::move(out));
            }
            return true;
        }
        if (m.err() == RdKafka::ERR__PARTITION_EOF) {
			return true;
		}
        std::cerr << "[WF-DP] Kafka source error: " << m.errstr() << "\n";
        return true;
    }
};


// Gestore del Flusso di Uscita Risultati verso Kafka
class KafkaSink_Functor {
	
public:
    explicit KafkaSink_Functor(const std::string &topic) : topic_(topic) {}
    wf::wf_kafka_sink_msg operator()(tuple_t &in) {
        wf::wf_kafka_sink_msg msg;
        msg.payload = in; 
		msg.topic = topic_;
        g_sink_msgs++;
        if (g_sink_msgs % 100 == 0) std::cerr << "[WF-DP] Sink emitted " << g_sink_msgs.load() << " messages\n";
        return msg;
    }
private:
    std::string topic_;
};


// Architetto della Topologia WindFlow
void run_pipeline(const CmdOptions& opts, const std::unordered_map<std::string, std::string>& props, const std::unordered_map<std::string, std::string>& exp_props) {
	
    using namespace wf;
    std::string bootstrap = props.count("kafka.bootstrap.servers") ? props.at("kafka.bootstrap.servers") : "";
    std::string input_topic = !opts.kafka_input_topic.empty() ? opts.kafka_input_topic : (props.count("kafka.input.data.topic") ? props.at("kafka.input.data.topic") : "");
    std::string output_topic = !opts.kafka_output_topic.empty() ? opts.kafka_output_topic : (props.count("kafka.output.topic") ? props.at("kafka.output.topic") : "");
   
    if (bootstrap.empty() || input_topic.empty() || output_topic.empty()) { std::cerr << "[WF-DP] ERROR: Missing Kafka config.\n"; return; }
   	
	int par = 1;
	auto it = exp_props.find("model_replicas");
	if (it != exp_props.end()) {
	    try {
	        par = std::max(1, std::stoi(it->second));
	    } catch (...) {
	        std::cerr << "[WF-DP] Invalid model_replicas = " << it->second << ", using 1\n";
	        par = 1;
	    }
	}
 
	std::cerr << "[WF-DP] experiment-config = " << opts.experiment_config << "\n";
	std::cerr << "[WF-DP] model_replicas = " << par << "\n";

    std::cout << "[WF-DP] Pipeline Config: Brokers = " << bootstrap << ", In =" << input_topic << ", Out =" << output_topic << ", Pm =" << opts.task_parallel << ", Par =" << par << "\n";

	int kafka_par = 2;
	it = props.find("kafka.input.data.partitions.num");
	if (it != props.end()) {
	    try {
	        kafka_par = std::max(1, std::stoi(it->second));
	    } catch (...) {
	        std::cerr << "[WF-DP] Invalid kafka partitions = " << it->second << ", using 2\n";
	        kafka_par = 1;
	    }
	}
	
	std::cerr << "[WF-DP] Kafka partitions = " << kafka_par << "\n"; // Kafka Partitions = 32
	
	int par_source = par; // Default N - N - N
	int par_infer  = par;
	int par_sink   = par;
	
	PipeGraph topology("wf_dp", Execution_Mode_t::DEFAULT, Time_Policy_t::INGRESS_TIME);
	
	if (opts.task_parallel == 't') {	// Task Parallel  32 - N - 32
		
	    par_source = kafka_par;
	    par_infer  = par;
	    par_sink   = kafka_par;
	    
	    std::cerr << "[WF-DP] Source Parallelism = " << par_source << "\n";
		std::cerr << "[WF-DP] Inference Functor Parallelism = " << par_infer << "\n";
		std::cerr << "[WF-DP] Sink Parallelism = " << par_sink << "\n";
		
	    KafkaSource_Functor source_functor;
	    Kafka_Source source = KafkaSource_Builder(source_functor)
						.withName("wf_kafka_source")
						.withBrokers(bootstrap)
						.withTopics(input_topic)
						.withGroupID("wf-dp-group-" + input_topic + "-" + std::to_string(::getpid()))
						.withAssignmentPolicy("roundrobin")
						.withIdleness(std::chrono::seconds(60))
						.withParallelism(par_source)
						.withOffsets(0)
						.build();
						
		Inference_Functor infer_functor(opts);
	    Map inference_op = Map_Builder([&infer_functor](const tuple_t& t) { return infer_functor(t); })
						.withName("wf_inference")
						.withParallelism(par_infer)
						.withRebalancing()
						.build();
	
	    KafkaSink_Functor sink_functor(output_topic);
	    Kafka_Sink sink = KafkaSink_Builder(sink_functor)
						.withName("wf_kafka_sink")
						.withParallelism(par_sink)
						.withRebalancing()
						.withBrokers(bootstrap)
						.build();
	
		topology.add_source(source)
				.add(inference_op)
				.add_sink(sink);
				
		std::cerr << "[WF-DP] Starting PipeGraph execution...\n";
	    topology.run();
	    std::cerr << "[WF-DP] PipeGraph finished.\n";
	
	} else {	// Data Parallel N - N - N
	
		par_source = par;
	    par_infer  = par;
	    par_sink   = par;
	    
	    std::cerr << "[WF-DP] Source Parallelism = " << par_source << "\n";
		std::cerr << "[WF-DP] Inference Functor Parallelism = " << par_infer << "\n";
		std::cerr << "[WF-DP] Sink Parallelism = " << par_sink << "\n";
	      
	    KafkaSource_Functor source_functor;
	    Kafka_Source source = KafkaSource_Builder(source_functor)
						.withName("wf_kafka_source")
						.withBrokers(bootstrap)
						.withTopics(input_topic)
						.withGroupID("wf-dp-group-" + input_topic + "-" + std::to_string(::getpid()))
						.withAssignmentPolicy("roundrobin")
						.withIdleness(std::chrono::seconds(60))
						.withParallelism(par_source)
						.withOffsets(0)
						.build();
						
		Inference_Functor infer_functor(opts);
	    Map inference_op = Map_Builder([&infer_functor](const tuple_t& t) { return infer_functor(t); })
						.withName("wf_inference")
						.withParallelism(par_infer)
						.build();
		
	    KafkaSink_Functor sink_functor(output_topic);
	    Kafka_Sink sink = KafkaSink_Builder(sink_functor)
						.withName("wf_kafka_sink")
						.withParallelism(par_sink)
						.withBrokers(bootstrap)
						.build();
		
		topology.add_source(source)
				.chain(inference_op)
				.chain_sink(sink);
   
	    std::cerr << "[WF-DP] Starting PipeGraph execution...\n";
	    topology.run();
	    std::cerr << "[WF-DP] PipeGraph finished.\n";
		
	}
	
}


// Main
int main(int argc, char** argv) {
	
    std::cout << "[WF-DP] WindFlow Data Processor starting..." << std::endl;
    
    CmdOptions opts = parse_args(argc, argv);
    
    std::cout << "[WF-DP] Model: " << opts.model_name << " (" << opts.model_format << ")\n";
    
    if (opts.global_config.empty()) {
		std::cerr << "[WF-DP] ERROR: global-config required.\n";
		return 1;
	}
	
    auto props = load_properties(opts.global_config);
	
	if (opts.experiment_config.empty()) {
		std::cerr << "[WF-DP] ERROR: experiment-config required.\n";
    	return 1;
	}
	
    auto exp_props = load_properties(opts.experiment_config);
	
    run_pipeline(opts, props, exp_props);
    
    std::cout << "[WF-DP] Shutdown." << std::endl;
    return 0;
}
