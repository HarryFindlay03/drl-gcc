/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/03/2024
 * FILE LAST UPDATED: 25/04/2024
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
    const std::vector<std::string>& program_names,
    const unsigned int buffer_size, 
    const unsigned int copy_period,
    const unsigned int number_of_episodes,
    const unsigned int episode_length,
    const double discount_rate,
    const double learning_rate,
    rand_helper* rnd,
    bool gradient_monitoring
)
:
    actions(actions), /* setting agent's action space */
    program_names(program_names),
    buffer_size(buffer_size), 
    copy_period(copy_period), 
    number_of_episodes(number_of_episodes), 
    episode_length(episode_length),
    discount_rate(discount_rate),
    learning_rate(learning_rate),
    rnd(rnd),
    gradient_monitoring(gradient_monitoring)
{
    // creating activation func pair and initialisor
    std::pair<mlp_activation_func_t, mlp_activation_func_t> activ_funcs = std::make_pair(mlp_ReLU, mlp_linear);
    weight_init_func_t initialiasor = he_normal_initialiser;
    mlp_loss_func_t loss_func = dql_square_loss_with_error_clipping;

    // instatiate both networks
    Q = new MLP(network_config, activ_funcs, initialiasor, loss_func, rnd, learning_rate);
    Q_hat = new MLP(network_config, activ_funcs, initialiasor, loss_func, rnd, learning_rate);

    // set optimisation baseline
    optimisation_baseline = "-O1"; // changeable parameter based on action-space chosen
    curr_env = construct_polybench_PolyString(program_names[0], optimisation_baseline);

    // initially set network weights equal
    copy_network_weights();

    // resize buffer and set curr pos
    buff.resize(buffer_size);
    curr_buff_pos = 0;

    // resize applied_optimisations to size of action space - 1 in pos i represents optimisation i has been applied
    applied_optimisations.resize(actions.size());

    // get no optimisations applied runtime
    init_runtime = run_given_string(curr_env->get_no_plugin_no_optimisations_PolyString(), program_names[0]);

    // open gradient file in order for agent to write gradient to file
    if(gradient_monitoring)
    {
        grad_monitor_file.open(GRADIENT_MONITOR_FILENAME);

        if(!grad_monitor_file.is_open())
        {
            std::cerr << "Gradient monitoring file open unsuccesful!\n";
            std::cerr << "Monitoring will not take place.\n";
        }
    }

    return;
}


void Agent::train_optimiser(const double epsilon)
{
    int i, j;
    bool terminate;
    int curr_itr = 0;

    for(i = 0; i < number_of_episodes; i++)
    {
        std::cout << "Episode: " << i << "\t Program: " << curr_env->program_name << "\t Training Progress: (" << ((i+1) / (double)number_of_episodes) * 100 << "%)\n" << std::flush;

        for(j = 0; j < episode_length; j++)
        {
            /* sampling */
            terminate = (((j+1) == episode_length) ? true : false);
            sampling(epsilon, terminate);

            /* sample from replay buffer and train */
            train_phase();

            /* copy network weights */
            if(!(curr_itr % copy_period))
                copy_network_weights();

            /* save network weights to file */
            if(!(curr_itr % ((int)DEFAULT_SAVE_PERIOD)))
                save_weights(Q, (std::string)DEFAULT_WEIGHT_SAVE_LOCATION);

            curr_itr++;
        }

        /* on episode completion */

        // output optimisations
        std::cout << "Optimisations applied in episode: ";
        for(auto const& opt : curr_env->optimisations)
            std::cout << opt << " ";
        std::cout << "\n\n" << std::flush;

        // reset the environment with a uniformally chosen new program, regenerate the initial runtime for the new chosen program, and reset applied optimisations
        int program_pos = rnd->random_int_range(0, program_names.size()-1);

        curr_env->reset_PolyString_environment(program_names[program_pos]);

        init_runtime = run_given_string(curr_env->get_no_plugin_no_optimisations_PolyString(), program_names[program_pos]);

        // reset applied_optimisations to all zeros
        for(auto it = applied_optimisations.begin(); it != applied_optimisations.end(); ++it)
            *it = 0;
    }
}


void Agent::sampling(const double epsilon, bool terminate)
{
    double reward = 0;
    std::vector<double> curr_st;
    std::vector<double> next_st;

    curr_st = vec_min_max_scaling(get_program_state(curr_env, get_num_features()));

    int action_pos = epsilon_greedy_action(curr_st, epsilon);

    // negatively reward if optimisation has already been applied and don't apply to the environment string 
    if(applied_optimisations[action_pos] == 1)
    {
        reward = -1;
    }
    else
    {
        // execute in emulator and observe reward
        curr_env->optimisations.push_back(actions[action_pos]);
        applied_optimisations[action_pos] = 1;
    }

    // get the next state after executing (applying) optimisation
    next_st = vec_min_max_scaling(get_program_state(curr_env, get_num_features()));

    // intermediate reward is zero if not episode termination else reward is proportional to the new program runtime compared
    // against the intitial runtime
    if(terminate)
    {
        double updt_runtime = run_given_string(curr_env->get_no_plugin_PolyString(), curr_env->program_name);
        reward = get_reward(updt_runtime);
        std::cout << "Initial Runtime:" << init_runtime << "\t New Runtime: " << updt_runtime << "\t Episode reward: " << reward << '\n';
    }

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

    // forward proporgate to save network output in Q object
    Eigen::MatrixXd out_Q = Q->forward_propogate(b->get_curr_st());

    // setting yj
    Eigen::MatrixXd out_yj = Eigen::MatrixXd::Zero(out_Q.rows(), out_Q.cols());
    out_yj(0, b->get_action_pos()) = y_j;

    if(gradient_monitoring)
    {
        grad_monitor_file << std::to_string((Q->loss_function(out_Q, out_yj, b->get_action_pos()))(0, b->get_action_pos())) << '\n';
        grad_monitor_file.flush();
    }

    // gradient descent step
    Q->back_propogate_rl(out_yj, b->get_action_pos());
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
    return DEFAULT_REWARD_FUNCTION(new_runtime, init_runtime);
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