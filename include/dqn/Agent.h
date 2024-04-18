/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 11/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: Class definition for deep Q-learning agent following DeepMind's paper.
*/

#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <map>

#include "mlp-cpp/network.h"
#include "mlp-cpp/funcs.h"

#include "utils/utils.h"
#include "utils/rand_helper.h"

#include "BufferItem.h"

#define NOP (std::string)""


/* HELPER FUNCTIONS */

/**
 * @brief squared loss function defined to be passed to ML_ANN
 * 
 * @param output 
 * @param target 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd l2_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);


std::vector<std::string> init_action_space(const std::vector<std::string>& as);


/* AGENT CLASS DEFINITION */


class Agent
{
    /* NETWORKS */
    MLP* Q;
    MLP* Q_hat;

    /* REPLAY BUFFER */
    std::vector<BufferItem*> buff;

    /* STARTING ACTION SPACE */
    std::vector<std::string> actions;

    /* CURRENT OPTIMISATIONS APPLIED */
    std::vector<int> applied_optimisations;

    /* UPDATED ENVIRONMENT CONTAINER */
    PolyString* curr_env;

    /* PARAMS */
    unsigned int buffer_size;
    unsigned int copy_period;
    unsigned int number_of_episodes;
    unsigned int episode_length;
    double discount_rate;
    double learning_rate;

    /* PRIVATE VALUES */
    double init_runtime;
    std::string program_name;
    unsigned int curr_buff_pos;

    /* RANDOM HELPER */
    rand_helper* rnd;


public:
    Agent
    (
        const std::vector<int>& network_config,
        const std::vector<std::string>& actions,
        const std::string& program_name,
        const unsigned int buffer_size, 
        const unsigned int copy_period,
        const unsigned int number_of_episodes,
        const unsigned int episode_length,
        const double discount_rate,
        const double learning_rate,
        rand_helper* rnd
    );

    ~Agent()
    {
        // delete each buffer item
        for(auto it = buff.begin(); it != buff.end(); ++it)
            delete *it;

        delete Q;
        delete Q_hat;
        delete rnd;
    };

    /* TRAINING FUNCTIONS */

    void train_optimiser(const double epsilon);

    void sampling(const double epsilon, bool terminate);

    void train_phase();

    /* HELPER FUNCTIONS*/

    /**
     * @brief Returns an index representing the chosen action from Agent's actions.
     * 
     * @param st 
     * @param epsilon 
     * @return int 
     */
    int epsilon_greedy_action(const std::vector<double>& st, const double epsilon);

    void copy_network_weights();

    inline const std::vector<std::string>& get_actions() { return actions; };

    double get_reward(const double new_runtime);

    inline double get_init_runtime() { return init_runtime; };

    int get_num_features() { return Q->layers[0]->W.rows(); };

    PolyString* get_PolyString() { return curr_env; }; // dangerous function

    /* DEBUG HELPER FUNCTIONS */

    void print_networks();

    /* STATIC HELPER FUNCTIONS */
    
    static int best_q_action(const Eigen::MatrixXd& input, int n);

};

#endif /* AGENT_H */