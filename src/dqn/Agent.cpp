/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 26/04/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: An implementation of a deep Q-learning agent following DeepMind's paper.
*/


#include "dqn/Agent.h"
#include "envtools/utils.h"


Agent::Agent
(
    const std::vector<size_t>& network_config,
    const std::vector<std::string>& action_space,
    const std::string unop_string,
    const std::string program_name,
    const unsigned int buffer_size,
    const unsigned int copy_period,
    const unsigned int number_of_episodes,
    const unsigned int episode_length
)
:
    unop_string(unop_string),
    program_name(program_name),
    buffer_size(buffer_size), 
    copy_period(copy_period), 
    number_of_episodes(number_of_episodes), 
    episode_length(episode_length)
{
    // instatiate both networks
    Q = new ML_ANN(network_config);
    Q_hat = new ML_ANN(network_config);

    // initially set network weights equal
    copy_network_weights();

    // resize buffer
    buff.resize(buffer_size);

    // set action space
    actions.resize(action_space.size());
    int pos = 0;
    for(auto it = actions.begin(); it != actions.end(); ++it)
        *it = action_space[pos++];     

    // get no optimisations applied runtime
    init_runtime = run_given_string((unop_string + "-O0"), program_name);

    return;
}


void Agent::train(const double epsilon)
{
    int i, j;
    
    for(i = 0; i < number_of_episodes; i++)
    {

        for(j = 0; j < episode_length; j++)
        {
            std::string action;

            // epsilon greedy select action from action space
            double r = ((double)rand() / (double)RAND_MAX);

            if(r > epsilon)
            {
                int pos = rand() % actions.size();
                action = actions[pos];

                // removing possible action from action space
                actions.erase(actions.begin() + pos);
            }
            else
            {
                // todo gather state of agent
                std::vector<double> st;

                Eigen::MatrixXd q_vals = Q->forward_propogate_rl(st);

                // choose action with best q values

                int best_pos = 0;
                int pos = 0;
                for(const auto& q : q_vals.rowwise())
                {
                    if(q[0] > q_vals.row(best_pos)[0])
                        best_pos = pos;

                    pos++;
                }

                action = actions[best_pos];
                actions.erase(actions.begin() + best_pos);
            }

            // running in emulator
            std::string new_compile_string = unop_string + action;
            double immediate_reward = get_reward(run_given_string(new_compile_string, program_name));

        }

    }
}


void Agent::copy_network_weights()
{
    // set Q_hat to Q (weights)

    // for each layer copy the weights matrix (excluding last)
    int i, j;
    for(i = 0; i < ((Q->get_num_layers())-1); i++)
        Q_hat->set_weight_matrix((Q->get_layers())[i]->W, i);

    return;
}


double Agent::get_reward(const double new_runtime)
{
    // return percentage difference of initial runtime vs new runtime
    if(new_runtime <= init_runtime)
        return 0;

    return (abs(init_runtime - new_runtime) /  ((init_runtime + new_runtime) / 2));
}