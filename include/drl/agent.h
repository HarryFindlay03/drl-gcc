#include "network.h"

typedef std::vector<std::vector<size_t> > replay_buffer_t;

class Agent
{
    ML_ANN* behaviour_net;
    ML_ANN* policy_net;
    int copy_period;
    int state_size;

    replay_buffer_t buffer;
    int buffer_size;

    std::vector<std::string> actions;

    std::vector<std::string> benchmarks;
    std::vector<std::string> optimsiations;

public:
    Agent(const std::vector<size_t>& network_config, const int buffer_size, const int copy_period, const std::vector<std::string>& actions);

    ~Agent()
    {
        delete behaviour_net;
        delete policy_net;
    }

    void train_agent(const std::string& compile_string, const int episodes, const double epsilon);

    void copy_network_weights();

    static void copy_network_weights(const ML_ANN& init_network);
};