
#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>

#include "Eigen/Core"

#include "utils/rand_helper.h"

/* helper typedefs */

typedef std::function<Eigen::MatrixXd(const Eigen::MatrixXd& input, bool deriv)> mlp_activation_func_t;

typedef std::function<void(Eigen::MatrixXd& mat, int fan_in, int fan_out, rand_helper* rnd)> weight_init_func_t;


/* main network class definitions */

class Layer
{
private:
    bool is_input;
    bool is_output;

public:

    /* MATRICES */

    Eigen::MatrixXd W; /* weight matrix */

    Eigen::MatrixXd S; /* input to layer */

    Eigen::MatrixXd Z; /* output from layer after activation function */

    Eigen::MatrixXd G; /* Gradient matrix for BP step */

    Eigen::MatrixXd Fp; /* Derivative of the activation function */

    /* ACTIVATION FUNCTION */

    mlp_activation_func_t activation_function;

public:

    Layer
    (
        int num_neurons,
        int num_next_neurons, 
        bool is_input, 
        bool is_output, 
        mlp_activation_func_t activation_function,
        rand_helper* rnd
    );

    Eigen::MatrixXd weighted_sum(double bias);

};


class MLP
{
public:

    std::vector<Layer*> layers; /* MLP layers */

    /* params */
    double learning_rate;

    /* misc */
    int num_layers;

public:
    MLP
    (
        const std::vector<int> &layer_config,
        const std::pair<mlp_activation_func_t, mlp_activation_func_t> &func_pair,
        weight_init_func_t initialiser,
        rand_helper *rnd,
        double learning_rate = 0.3
    );

    /* main network functions */
    Eigen::MatrixXd forward_propogate(const std::vector<double>& input);

    void back_propogate(const Eigen::MatrixXd& target);

    void update_weights();
};

#endif /* NETWORK_H */