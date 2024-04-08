/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 08/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: An implementation of a deep Q-learning agent following DeepMind's paper.
*/

/* TODO */
    // implement global static random number generation


#include "dqn/Agent.h"


/* HELPER FUNCTIONS */


Eigen::MatrixXd l2_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    Eigen::MatrixXd diff = (output-target);

    int i;
    for(i = 0; i < diff.size(); i++)
        *(diff.data() + i) = std::pow(*(diff.data() + i), 2);

    return diff;
}


/* AGENT CLASS*/


Agent::Agent
(
    const std::vector<size_t>& network_config,
    const std::vector<std::string>& action_space,
    const std::string unop_string,
    const std::string program_name,
    const unsigned int buffer_size,
    const unsigned int copy_period,
    const unsigned int number_of_episodes,
    const unsigned int episode_length,
    const double discount_rate,
    const double eta
)
:
    unop_string(unop_string),
    program_name(program_name),
    buffer_size(buffer_size), 
    copy_period(copy_period), 
    number_of_episodes(number_of_episodes), 
    episode_length(episode_length),
    discount_rate(discount_rate),
    eta(eta)
{
    // instatiate both networks
    Q = new ML_ANN(network_config, l2_loss);
    Q_hat = new ML_ANN(network_config, l2_loss);

    // initially set network weights equal
    copy_network_weights();

    // resize buffer and set curr pos
    buff.resize(buffer_size);
    curr_buff_pos = 0;

    // set action space
    actions.resize(action_space.size());
    int pos = 0;
    for(auto it = actions.begin(); it != actions.end(); ++it)
        *it = action_space[pos++];     

    // get no optimisations applied runtime
    init_runtime = run_given_string((unop_string + "-O0"), program_name);

    // instatiante random generator
    rnd = new RandHelper();

    return;
}


void Agent::train_optimiser(const double epsilon)
{
    int i, j, curr_itr;
 
    for(i = 0, curr_itr = 0; i < number_of_episodes; i++)
    {

        for(j = 0; j < episode_length; j++)
        {
            std::string action;
            
            // states
            std::vector<double> curr_st;
            std::vector<double> next_st;

            curr_st = get_program_state((unop_string + opt_vec_to_string(applied_optimisations)));
            action = Agent::epsilon_greedy_action<std::string>(Q, actions, curr_st, rnd, epsilon);

            /* execute action a_t in emulator and and observe reward r_t and image x_(t+1)*/
            // running in emulator - returning the immediate reward here but what is that ?
            std::string new_compile_string = unop_string + action;

            // the reward is the value returned from the network with parameters theta?

            // xiaoyang - intermediate reward is zero (unless at episode end)
            double reward;

            /* set s_(t+1) and preprocess - preprocessing is anagalous to getting the state vector for input into NN */
            next_st = get_program_state((unop_string + opt_vec_to_string(applied_optimisations) + action));

            /* store the transistion in the replay buffer D (preprocessed s_t, a_t, r_t, preprocessed s_(t+1)) */
            // BufferItem* trans = new BufferItem(curr_st, action, reward, next_st);
            buff[(curr_buff_pos++) % buffer_size] = new BufferItem(curr_st, action, reward, next_st);


            /* sample a transition from D */
            BufferItem* sample = buff[rnd->random_int_range(0, buff.size())];

            double sample_reward = sample->get_reward();
            if((j+1) != (episode_length-1))
            {
                sample_reward += (discount_rate * best_q_hat_value(sample->get_next_st()));
            }

            auto sample_actions_avail = sample->get_actions_avail();
            int action_pos = (std::find(sample_actions_avail.begin(), sample_actions_avail.end(), sample->get_action())) - sample_actions_avail.begin();

            if((action_pos == sample_actions_avail.size()) && (sample_actions_avail[sample_actions_avail.size()-1] != sample->get_action()))
            {
                std::cout << "ERROR: Action in state not in states available actions, exiting function." << std::endl;
                return;
            }

            Eigen::MatrixXd output_vec = Q->forward_propogate_rl(sample->get_curr_st());
            output_vec(action_pos, 0) += sample_reward;

            Q->back_propogate_rl(output_vec, Q->forward_propogate_rl(sample->get_curr_st()));
            Q->update_weights_rl(eta);

            if(curr_itr % copy_period == 0)
                copy_network_weights();
        }
    }
}


void Agent::copy_network_weights()
{
    // set Q_hat to Q (weights)

    // for each layer copy the weights matrix (excluding last)
    int i;
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


double Agent::best_q_hat_value(const std::vector<double>& new_env_st)
{
    // find the best q_val for Q_hat output given the new state

    Eigen::MatrixXd q_hat_vals = Q_hat->forward_propogate_rl(new_env_st);

    double best_v = -1;
    for(const auto & v : q_hat_vals.rowwise())
    {
        if(v[0] > best_v)
            best_v = v[0];
    }

    return best_v;
}


void Agent::print_networks()
{
    std::cout << "Q network:\n";

    for(auto l : Q->get_layers())
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    std::cout << "Q_hat network:\n";

    for(auto l : Q_hat->get_layers())
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    return;
}


/* STATIC HELPER FUNCTIONS */


template <typename T>
T Agent::epsilon_greedy_action(ML_ANN* Q, std::vector<T>& actions, const std::vector<double>& st, RandHelper* rnd, double epsilon)
{
    T action_res;

    // double r = ((double)rand() % (double)RAND_MAX);
    double r = rnd->random_double_range(0.0, 1.0);

    if(r > epsilon)
    {
        int pos = rand() % actions.size();
        action_res = actions[pos];

        // removing action from action space
        actions.erase(actions.begin() + pos);

        return action_res;
    }

    Eigen::MatrixXd q_vals = Q->forward_propogate_rl(st);

    int best_pos = 0;
    int pos = 0;
    for (const auto &r : q_vals.rowwise())
    {
        if (r[0] > q_vals.row(best_pos)[0])
            best_pos = pos;
        pos++;
    }

    action_res = actions[best_pos];
    actions.erase(actions.begin() + best_pos);

    return action_res;
}


