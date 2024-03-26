/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 25/04/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: Class definition for deep Q-learning agent following DeepMind's paper.
*/

#include <string>
#include <vector>
#include <random>

#include "cpp-nn/network.h"
#include "BufferItem.h"


class Agent
{
    /* NETWORKS */
    ML_ANN* Q;
    ML_ANN* Q_hat;

    /* REPLAY BUFFER */
    std::vector<BufferItem*> buff;

    /* STARTING ACTION SPACE */
    std::vector<std::string> actions;

    /* UNOPTIMISED STRING - acts as environment */
    std::string unop_string;

    /* PARAMS */
    unsigned int buffer_size;
    unsigned int copy_period;
    unsigned int number_of_episodes;
    unsigned int episode_length;

    /* PRIVATE VALUES */
    double init_runtime;
    std::string program_name;

    // state extraction function
    // loss function - passed to cpp-nn!


public:
    Agent
    (
        const std::vector<size_t>& network_config,
        const std::vector<std::string>& actions,
        const std::string unop_string,
        const std::string program_name,
        const unsigned int buffer_size, 
        const unsigned int copy_period,
        const unsigned int number_of_episodes,
        const unsigned int episode_length
    );

    ~Agent()
    {
        // delete each buffer item
        for(auto it = buff.begin(); it != buff.end(); ++it)
            delete *it;

        delete Q;
        delete Q_hat;
    };

    void train(const double epsilon);

    void copy_network_weights();

    double get_reward(const double new_runtime);
};