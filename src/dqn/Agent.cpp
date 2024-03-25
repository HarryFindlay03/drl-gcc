#include "dqn/Agent.h"

Agent::Agent
(
    const std::vector<size_t>& network_config,
    const unsigned int buffer_size,
    const unsigned int copy_period,
    const unsigned int number_of_episodes,
    const unsigned int episode_length
)
:
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

    return;
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