#include "dqn/Agent.h"

#define MY_SEED 12349

int main()
{
    // input layer - num features parsed in statetool
    // output layer - size of action space

    std::vector<std::string> actions = read_file_to_vec("data/action_spaces/LOOPS_CSE_actionspace.txt");
    actions.push_back(NOP);

    int output_layer_size = actions.size();

    std::string program_name = "jacobi-2d-imper";

    std::vector<int> network_config = {7, 30, 30, 30, output_layer_size};

    MLP* policy_net = new MLP(network_config, std::make_pair(DEFAULT_HIDDEN_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION), DEFAULT_INITIALISOR, DEFAULT_LOSS_FUNCTION, new rand_helper(MY_SEED), 0.001);
    load_weights(policy_net, "data/saved_weights/300.txt");

    std::vector<std::string> chosen_opts = Agent::select_actions_via_policy(policy_net, program_name, actions, "-O1", 6);

    std::cout << "Program name: " << program_name << std::endl;
    std::cout << "CHOSEN OPTIMISATIONS: ";
    for(const auto & opt : chosen_opts)
        std::cout << (opt + " ");
    std::cout << std::endl;

    return 0;
}