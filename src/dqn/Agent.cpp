/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 09/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: An implementation of a deep Q-learning agent following DeepMind's paper.
*/


#include "dqn/Agent.h"


/* AGENT CLASS*/


Agent::Agent
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
)
:
    actions(actions), /* setting agent's action space */
    program_name(program_name),
    buffer_size(buffer_size), 
    copy_period(copy_period), 
    number_of_episodes(number_of_episodes), 
    episode_length(episode_length),
    discount_rate(discount_rate),
    learning_rate(learning_rate),
    rnd(rnd)
{

    // creating activation func pair and initialisor
    std::pair<mlp_activation_func_t, mlp_activation_func_t> activ_funcs = std::make_pair(mlp_ReLU, mlp_linear);
    weight_init_func_t initialiasor = he_normal_initialiser;
    mlp_loss_func_t loss_func = dql_square_loss;

    // instatiate both networks
    Q = new MLP(network_config, activ_funcs, initialiasor, loss_func, rnd, learning_rate);
    Q_hat = new MLP(network_config, activ_funcs, initialiasor, loss_func, rnd, learning_rate);

    // construct agent's PolyString (environnment)
    curr_env = construct_polybench_PolyString(program_name);

    // initially set network weights equal
    copy_network_weights();

    // resize buffer and set curr pos
    buff.resize(buffer_size);
    curr_buff_pos = 0;

    // get no optimisations applied runtime
    init_runtime = run_given_string(curr_env->get_no_plugin_no_optimisations_PolyString(), program_name);

    return;
}


void Agent::train_optimiser(const double epsilon)
{
    int i, j;
    bool terminate;
    int curr_itr = 0;

    for(i = 0; i < number_of_episodes; i++)
    {
        // reset polystring
        curr_env->reset_PolyString_optimisations();

        for(j = 0; j < episode_length; j++)
        {
            /* sampling */
            terminate = (((j+1) == episode_length) ? true : false);
            sampling(epsilon, terminate);

            /* sample from replay buffer and train */
            train_phase();

            /* copy network weights */
            if((curr_itr++) % copy_period)
                copy_network_weights();
        }
    }
}


void Agent::sampling(const double epsilon, bool terminate)
{
    double reward = 0;
    std::vector<double> curr_st;
    std::vector<double> next_st;

    curr_st = get_program_state(curr_env, get_num_features());

    int action_pos = epsilon_greedy_action(curr_st, epsilon);

    // negatively reward if optimisation has already been applied and don't apply to the environment string 
    if(std::find(applied_optimisations.begin(), applied_optimisations.end(), action_pos) != applied_optimisations.end())
    {
        reward = -1;
    }
    else
    {
        // execute in emulator and observe reward
        curr_env->optimisations.push_back(actions[action_pos]);
    }

    // add to optimisations applied

    next_st = get_program_state(curr_env, get_num_features());

    // intermediate reward is zero if not episode termination else reward is proportional to the new program runtime compared
    // against the intitial runtime
    if(terminate)
        reward = get_reward(run_given_string(curr_env->get_no_plugin_PolyString(), program_name));

    // save to replay buffer
    buff[(curr_buff_pos++) % buffer_size] = new BufferItem(curr_st, action_pos, reward, next_st, terminate);

    return;
}


void Agent::train_phase()
{
    /* uniformly sample the replay buffer */
    int max_size = (buff[(curr_buff_pos) % buffer_size] == NULL) ? curr_buff_pos : buffer_size;
    BufferItem* b = buff[rnd->random_int_range(0, max_size - 1)];


    double y_j;
    if(b->get_terminate())
    {
        y_j = b->get_reward();
    }
    else
    {
        // find the best action value with Q_hat
        Eigen::MatrixXd out = Q_hat->forward_propogate(b->get_next_st());

        int best_pos = Agent::best_q_action(out, actions.size());

        y_j = b->get_reward() + (discount_rate * out(0, best_pos));
    }

    // gradient descent step only on output node j for action j.
    Eigen::MatrixXd out_Q = Q->forward_propogate(b->get_curr_st());

    // need the action pos of j - this is where we set yj
    Eigen::MatrixXd out_yj = out_Q;
    out_yj(0, b->get_action_pos()) = y_j;

    // gradient descent step
    Q->back_propogate(out_yj);
    Q->update_weights();

    return;    
}


/* HELPER FUNCTIONS */


int Agent::epsilon_greedy_action(const std::vector<double>& st, const double epsilon)
{
    double r = rnd->random_double_range(0.0, 1.0);

    if(r > (1 - epsilon))
    {
        return rnd->random_int_range(0, actions.size()-1);
    }

    Eigen::MatrixXd q_vals = Q->forward_propogate(st);

    int best_pos = Agent::best_q_action(q_vals, actions.size());

    return best_pos;
}


void Agent::copy_network_weights()
{
    // set Q_hat to Q (weights)

    // for each layer copy the weights matrix (excluding last)
    int i;
    for(i = 0; i < ((Q->num_layers)-1); i++)
        Q_hat->layers[i]->W = Q->layers[i]->W; // this may be weird

    return;
}


double Agent::get_reward(const double new_runtime)
{
    // return percentage difference of initial runtime vs new runtime
    if(new_runtime <= init_runtime)
        return 0;

    return (abs(init_runtime - new_runtime) /  ((init_runtime + new_runtime) / 2));
}


void Agent::print_networks()
{
    std::cout << "Q network:\n";

    for(auto l : Q->layers)
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    std::cout << "Q_hat network:\n";

    for(auto l : Q_hat->layers)
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    return;
}


/* STATIC HELPER FUNCTIONS */


int Agent::best_q_action(const Eigen::MatrixXd& input, int n)
{
    int best_pos = 0;
    int i;
    for(i = 1; i < n; i++)
    {
        if(input(0, i) > input(0, best_pos))
            best_pos = i;
    }

    return best_pos;
}