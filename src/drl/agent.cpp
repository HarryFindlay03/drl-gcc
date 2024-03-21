#include "agent.h"
#include "utils.h"


/* AGENT CLASS IMPLEMENTATION */

Agent::Agent(const std::vector<size_t>& network_config, const int buffer_size, int copy_period, const std::vector<std::string>& actions)
    : buffer_size(buffer_size), copy_period(copy_period), actions(actions)
{
    // read benchmarks and optimisations
    read_file_to_vec(benchmarks, DEFAULT_BENCHMARKS_LIST_LOCATION);
    read_file_to_vec(optimsiations, DEFAULT_OPTIMISATIONS_LIST_LOCATION);

    // init behaviour network and policy network the same as network_config
    behaviour_net = new ML_ANN(network_config);
    policy_net = new ML_ANN(network_config);
    state_size = network_config[0];

    // copy network weights to ensure that they are the same
    copy_network_weights();

    // sizing buffer and each state in buffer
    buffer.resize(buffer_size);
}


void Agent::train_agent(const std::string& benchmark, const int episodes, const double epsilon)
{
    // first get unoptimised runtime to gather final reward - working with polybench currently
    std::string benchmark_string = format_benchmark_string(benchmark, benchmarks);
    double init_runtime = run_given_string((benchmark_string + "-O0"), benchmark);

    // run for episodes
    int i, j;
    for(i = 0; i < episodes; i++)
    {
        std::vector<size_t> curr_state;
        // todo get program state
            // run the compiler with in development gcc plugin enabled, gather the program state from location

        // epsilon greedy select action by feeding current state to network
        double rand; // random number between 0 and 1
        std::string action_to_exec;
        if(rand > (1-epsilon))
        {
            // randomly select action
            int random_pos; // range(0, actions.size()-1)
            action_to_exec = actions[random_pos];
        }
        else
        {
            std::vector<double> q_vals(actions.size());
            int pos = 0;
            for (const auto &a : actions)
            {
                // get current state of action
                std::vector<size_t> curr_state;

                // normalisation step of curr_state

                std::vector<double> norm_curr_state;
                q_vals[pos++] = policy_net->forward_propogate_rl(norm_curr_state);
            }

            int best_pos = 0;
            for (j = 1; j < q_vals.size(); j++)
                if(q_vals[j] > q_vals[best_pos])
                    best_pos = j;

            action_to_exec = actions[best_pos];            
        }
        
        // execute the action in the environment

        // get the new state
        std::vector<size_t> new_state;

        // append the current state to the buffer
        buffer[i % buffer_size] = curr_state;

        // experience replay - randomly sampling from the buffer

        // take target batch or Qt
        // gradient descent on Q

        // if weight update frequency then copy weights

    }
}


void Agent::copy_network_weights()
{
    // copying from behaviour network to policy network
    int i;
    for(i = 0; i < behaviour_net->get_num_layers(); i++)
        policy_net->layers[i]->W = behaviour_net->layers[i]->W;

    return;
}