#include <iostream>

#include "mlp-cpp/funcs.h"
#include "mlp-cpp/network.h"

#define MY_RANDOM_SEED 14264


struct DataHolder
{
    double x;
    double sin_x;

    DataHolder(double x, double sin_x) : x(x), sin_x(sin_x) {};
};


bool check_double(double lhs, double rhs)
{
    return std::fabs(lhs - rhs) < 1E-02;
}


int main(void)
{
    int i;
    rand_helper* rnd = new rand_helper(MY_RANDOM_SEED);

    std::cout << "\n\nNEURAL NETWORK TESTING\n\n";

    std::vector<int> layer_config = {1, 30, 30, 30, 1};
    std::pair<mlp_activation_func_t, mlp_activation_func_t> func_pair = std::make_pair(mlp_sigmoid, mlp_linear);
    weight_init_func_t initialiser = xiaver_initialiser;
    mlp_loss_func_t loss_function = standard_loss;
    double learning_rate = 0.1;

    MLP* mlp = new MLP(layer_config, func_pair, initialiser, loss_function, rnd, learning_rate);

    double accuracy = 0.5;
    int size = 13;
    double x_val = 0;
    std::vector<DataHolder*> train_data(size);

    for(i = 0; i < size; i++, x_val += accuracy)
    {
        train_data[i] = new DataHolder(x_val, sin(x_val));
    }

    int train_epochs = 30000;
    int correct = 0;
    int j;
    for(i = 0; i < size * train_epochs; i++)
    {
        int pos = i % size;
        auto out = mlp->forward_propogate({train_data[pos]->x});

        Eigen::MatrixXd tar(1, 1);
        tar << train_data[pos]->sin_x;

        std::cout << "Network output: " << out << "\t Expected output: " << tar << '\n';
        if(check_double(out(0, 0), tar(0, 0)))
            correct++;

        std::cout << "Network Accuracy: " << ((double)correct / (i + 1)) * 100 << "%\n";

        mlp->back_propogate(tar);
        mlp->update_weights();
    }

    // saving weights
    save_weights(mlp, "example_sin_weights.txt");

    // loading weights
    MLP* t_mlp = new MLP(layer_config, func_pair, initialiser, loss_function, rnd, learning_rate);
    load_weights(t_mlp, "example_sin_weights.txt");

    // testing with loaded weights
    for(i = 0, correct = 0; i < size; i++)
    {
        int pos = i;
        auto out = mlp->forward_propogate({train_data[pos]->x});

        Eigen::MatrixXd tar(1, 1);
        tar << train_data[pos]->sin_x;

        std::cout << "(LOADED WEIGHTS) Network output: " << out << "\t Expected output: " << tar << '\n';
        if(check_double(out(0, 0), tar(0, 0)))
            correct++;

        std::cout << "(LOADED WEIGHTS) Network Accuracy: " << ((double)correct / (i + 1)) * 100 << "%\n";
    }


    return 0;
}

