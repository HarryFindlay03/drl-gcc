#include "dqn/Agent.h"

#define MY_SEED 1234

int main()
{
    std::vector<int> network_config = {12, 12, 12, 5};
    std::vector<std::string> actions = {"-fsplit-loops", "-ftree-loop-distribution", "-fvect-cost-model=dynamic", "-funswitch-loops", "-fversion-loops-for-strides"};
    std::string program_name = "covariance";

    unsigned int buffer_size = 10;
    unsigned int copy_period = 4;
    unsigned int number_episodes = 10;
    unsigned int episode_length = 3;
    double discount_rate = 0.9;
    double learning_rate = 0.1;

    rand_helper* rnd = new rand_helper(MY_SEED);

    Agent* ag = new Agent(network_config, actions, program_name, buffer_size, copy_period, number_episodes, episode_length, discount_rate, learning_rate, rnd);

    ag->print_networks();

    // auto t_unop_state_vec = get_program_state(ag->get_PolyString(), 12);

    // for(auto const& v : t_unop_state_vec)
    //     std::cout << v << " ";
    // std::cout << std::endl;

    // std::cout << "Initial runtime: " << ag->get_init_runtime() << '\n';

    // double epsilon = 0.3;
    // ag->train_optimiser(epsilon);

    return 0;
}