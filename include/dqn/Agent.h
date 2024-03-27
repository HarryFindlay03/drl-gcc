/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 27/04/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: Class definition for deep Q-learning agent following DeepMind's paper.
*/

#include <string>
#include <vector>
#include <cmath>
#include <random>

#include "cpp-nn/network.h"
#include "envtools/utils.h"
#include "envtools/RandHelper.h"
#include "BufferItem.h"


/* HELPER FUNCTIONS */

/**
 * @brief squared loss function defined to be passed to ML_ANN
 * 
 * @param output 
 * @param target 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd l2_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);


/* AGENT CLASS DEFINITION */


class Agent
{
    /* NETWORKS */
    ML_ANN* Q;
    ML_ANN* Q_hat;

    /* REPLAY BUFFER */
    std::vector<BufferItem*> buff;

    /* STARTING ACTION SPACE */
    std::vector<std::string> actions;

    /* CURRENT OPTIMISATIONS APPLIED */
    std::vector<std::string> applied_optimisations;

    /* UNOPTIMISED STRING - acts as environment */
    std::string unop_string;

    /* PARAMS */
    unsigned int buffer_size;
    unsigned int copy_period;
    unsigned int number_of_episodes;
    unsigned int episode_length;
    double discount_rate;
    double eta;

    /* PRIVATE VALUES */
    double init_runtime;
    std::string program_name;
    unsigned int curr_buff_pos;
    RandHelper* rnd;

    // state extraction function
    // loss function - passed to cpp-nn!

    /**
     * @brief finds and returns the best q value for q_hat
     * 
     * @return double 
     */
    inline double best_q_hat_value(const std::vector<double>& new_env_st);


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
        const unsigned int episode_length,
        const double discount_rate,
        const double eta
    );

    ~Agent()
    {
        // delete each buffer item
        for(auto it = buff.begin(); it != buff.end(); ++it)
            delete *it;

        delete Q;
        delete Q_hat;
    };

    void train_optimiser(const double epsilon);

    void copy_network_weights();

    double get_reward(const double new_runtime);


    /* STATIC HELPER FUNCTIONS */

    template <typename T>
    static T epsilon_greedy_action(const ML_ANN* Q, std::vector<T>& actions, const std::vector<double>& st, const double epsilon);

};