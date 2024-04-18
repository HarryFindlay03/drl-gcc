
#include "mlp-cpp/network.h"

/* LAYER CLASS IMPLEMENTATION */

Layer::Layer
(
    int num_neurons,
    int num_next_neurons,
    bool is_input,
    bool is_output,
    mlp_activation_func_t activation_function,
    rand_helper* rnd
)
: is_input(is_input), is_output(is_output), activation_function(activation_function)
{
    // input, hidden layers and output layers
    Z = Eigen::MatrixXd::Zero(1, num_neurons);

    // hidden layers and input layer
    if(!is_output)
    {
        W = Eigen::MatrixXd::Zero(num_neurons, num_next_neurons);
    }

    // hidden layers and output layer
    if(!is_input)
    {
        S = Eigen::MatrixXd::Zero(1, num_neurons);
        G = Eigen::MatrixXd::Zero(1, num_neurons);
        Fp = Eigen::MatrixXd::Zero(1, num_neurons);
    }
}


Eigen::MatrixXd Layer::weighted_sum(double bias)
{
    if(!is_input)
    {
        // store deriv of activation func with layer input for later
        Fp = activation_function(S, true);

        // run layer input through activation function
        Z = activation_function(S, false);
    }

    // not weighted sum if output layer - we just want to compute the above values
    if(is_output)
        return Z;

    // computing bias matrices
    Eigen::MatrixXd Z_bias = Z;
    Z_bias.conservativeResize(Z_bias.rows(), Z_bias.cols()+1);
    Z_bias.col(Z_bias.cols()-1) = Eigen::MatrixXd::Constant(1, 1, 1);

    Eigen::MatrixXd W_bias = W;
    W_bias.conservativeResize(W_bias.rows()+1, W_bias.cols());
    W_bias.row(W_bias.rows()-1) = Eigen::MatrixXd::Constant(1, W_bias.cols(), bias);

    // weighted sum of inputs using computed bias matrices
    return (Z_bias * W_bias);
}


/* MLP CLASS IMPLEMENTATION */


MLP::MLP
(
    const std::vector<int>& layer_config,
    const std::pair<mlp_activation_func_t, mlp_activation_func_t> & func_pair,
    weight_init_func_t initialiser,
    rand_helper* rnd,
    double learning_rate
)
: learning_rate(learning_rate)
{
    num_layers = layer_config.size();

    // resizing layers vector
    layers.resize(num_layers);

    // setting output layer
    layers[0] = new Layer(layer_config[0], layer_config[1], true, false, NULL, rnd);

    // setting hidden layers
    int i;
    for(i = 1; i < (num_layers - 1); i++)
        layers[i] = new Layer(layer_config[i], layer_config[i+1], false, false, std::get<0>(func_pair), rnd);

    // setting output layer
    layers[num_layers-1] = new Layer(layer_config[num_layers-1], 0, false, true, std::get<1>(func_pair), rnd);

    // initialising the weights
    if(initialiser != NULL)
    {
        for(i = 0; i < num_layers-1; i++)
        {
            int fan_in = (i == 0) ? 1 : layers[i-1]->W.rows();
            int fan_out = layers[i]->W.cols();
            initialiser(layers[i]->W, fan_in, fan_out, rnd);
        }
    }
    else // uniformally randomise the weights
    {
        int x, y;
        for(i = 0; i < num_layers-1; i++)
        {
            for(x = 0; x < layers[i]->W.rows(); x++)
            {
                for(y = 0; y < layers[i]->W.cols(); y++)
                    (layers[i]->W)(x, y) = rnd->random_double_range(-1.0, 1.0);
            }
        }
    }
}


Eigen::MatrixXd MLP::forward_propogate(const std::vector<double>& input)
{
    double bias = 0.3; // todo make dynamic

    // ensure input and first layer are of same dimension
    int n = layers[0]->Z.cols();
    if(n != input.size())
    {
        std::cerr << "Data input to network is not of correct size, or network layout has been set incorrectly, exiting!\n";
        std::exit(-1);
    }

    // set Z in the input layer - no activation function
    int i;
    for(i = 0; i < n; i++)
        (layers[0]->Z)(0, i) = input[i];

    // work out the input to the remaining layers as the weighted sum of the ouptut of previous layer
    for(i = 1; i < layers.size(); i++)
    {
        layers[i]->S = layers[i-1]->weighted_sum(bias);
    }

    // compute Fp and Z for output using weighted_sum function, this returns early - does not compute a weighted sum.
    return layers[num_layers-1]->weighted_sum(bias);
}


void MLP::back_propogate(const Eigen::MatrixXd& target)
{
    /* output layer */
    layers[num_layers-1]->G = (target - layers[num_layers-1]->Z);

    /* back propogating through remaining excluding input */
    int i;
    for(i = (num_layers-2); i > 0; i--)
        layers[i]->G = layers[i]->Fp.array().eval() * (layers[i+1]->G * layers[i]->W.transpose().eval()).array().eval();
}


void MLP::update_weights()
{
    int i;
    for(i = 0; i < (num_layers-1); i++)
    {
        Eigen::MatrixXd W_diff = (learning_rate) * (layers[i]->Z.transpose().eval() * layers[i+1]->G);
        layers[i]->W += W_diff;
    }
}