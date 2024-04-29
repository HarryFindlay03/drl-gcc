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
    save_agent_information();

    // creating activation func pair and initialisor
    std::pair<mlp_activation_func_t, mlp_activation_func_t> activ_funcs = std::make_pair(DEFAULT_HIDDEN_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION);
    weight_init_func_t initialiasor = DEFAULT_INITIALISOR;
    mlp_loss_func_t loss_func = DEFAULT_LOSS_FUNCTION;

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
        std::cout << "Episode: " << i << "\t Program: " << curr_env->program_name << "\t Training Progress: " << ((i+1) / (double)number_of_episodes) * 100 << "%\n" << std::flush;

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

    /* on completion save weights */
    save_weights(Q, (std::string)DEFAULT_WEIGHT_SAVE_LOCATION);
    std::cout << "Training complete, weights saved to location: " << DEFAULT_WEIGHT_SAVE_LOCATION << '\n' << std::flush;

    print_agent_information();

    return;
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
        grad_monitor_file << std::to_string((Q->loss_function(out_yj, out_Q, b->get_action_pos()))(0, b->get_action_pos())) << '\n';
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


void Agent::print_agent_information()
{
    std::cout << "=================\n" << "PARAMETERS\n" << "=================\n";
    std::cout << "Learning rate: " << learning_rate << '\n';
    std::cout << "Number of episodes: " << number_of_episodes << '\n';
    std::cout << "Episode length: " << episode_length << '\n';
    std::cout << "Discount rate: " << discount_rate << '\n';
    std::cout << "Buffer size: " << buffer_size << '\n';
    std::cout << "\n=================\n" << "TRAINING INFORMATION\n" << "=================\n";
    std::cout << "Action space: " << opt_vec_to_string(actions) << '\n';
    std::cout << "\nProgram training space: " << opt_vec_to_string(program_names) << '\n';
    std::cout << std::endl;

    return;
}


void Agent::save_agent_information()
{
    std::ofstream out_file(DEFAULT_AGENT_INFO_LOCATION);

    if(!(out_file.is_open()))
    {
        std::cerr << "ERROR IN SAVING AGENT INFORMATION, THIS WILL BE PRINTED STILL AT THE END OF PROGRAM RUN!\n";
        return;
    }

    out_file << "=================\n" << "PARAMETERS\n" << "=================\n";
    out_file << "Learning rate: " << learning_rate << '\n';
    out_file << "Number of episodes: " << number_of_episodes << '\n';
    out_file << "Episode length: " << episode_length << '\n';
    out_file << "Discount rate: " << discount_rate << '\n';
    out_file << "Buffer size: " << buffer_size << '\n';
    out_file << "\n=================\n" << "TRAINING INFORMATION\n" << "=================\n";
    out_file << "Action space: " << opt_vec_to_string(actions) << '\n';
    out_file << "\nProgram training space: " << opt_vec_to_string(program_names) << '\n';
    out_file << std::endl;

    out_file.close();

    return;
}


/* STATIC TRAINED POLICY FUNCTIONS */


std::vector<std::string> Agent::select_actions_via_policy(MLP* Q_net, const std::string& program_name, const std::vector<std::string>& action_space, const std::string& optimisation_baseline, int num_actions)
{
    std::vector<int> selected(action_space.size());
    std::vector<double> curr_st;

    // generate agent's environment
    PolyString* curr_env = construct_polybench_PolyString(program_name, optimisation_baseline);

    int i;
    for(i = 0; i < num_actions; i++)
    {
        // forward prop the curr_env to get q_vals
        curr_st = vec_min_max_scaling(get_program_state(curr_env, Q_net->layers[0]->W.rows()));

        Eigen::MatrixXd vals = Q_net->forward_propogate(curr_st);

        int best_pos = best_q_action(vals, vals.cols());

        // append to ret vector only if not already selected
        if(selected[best_pos] != 1)
        {
            selected[best_pos] = 1;
            curr_env->optimisations.push_back(action_space[best_pos]);
        }
    }

    std::vector<std::string> ret(curr_env->optimisations);
    return ret;
}


std::vector<std::string> Agent::select_actions_via_policy(MLP* Q_net, const std::string& program_name, const std::vector<std::string>& action_space, const std::string& optimisation_baseline)
{
    return select_actions_via_policy(Q_net, program_name, action_space, optimisation_baseline, action_space.size());
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