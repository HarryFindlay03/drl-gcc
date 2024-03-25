#ifndef LAYER_H
#define LAYER_H

#define RANDOM_SEED 12345

class Layer
{
    bool is_input, is_output;

    std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func;


public:
    static int count;
    static std::default_random_engine random_engine;

    Eigen::MatrixXd Z; // holds output values;
    Eigen::MatrixXd S; // output values pre activation function - "inputs into the layer"
    Eigen::MatrixXd W; // outgoing weight matrix for layer
    Eigen::MatrixXd Fp; // holds the derivatives for this layer
    Eigen::MatrixXd G; // gradient matrix


    Layer(int curr_size, int next_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func);


    ~Layer() {} 


    Eigen::MatrixXd forward_propogate_rl();
};

#endif