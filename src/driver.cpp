#include "cpp-nn/network.h"

#include <vector>

/* PREPPING THE RANDOM ENGINE AND LAYER COUNT */
std::default_random_engine Layer::random_engine;
int Layer::count = 0;

int main()
{

    std::vector<size_t> network_config = {3, 3, 3, 3};
    ML_ANN* Q = new ML_ANN(network_config);

    std::vector<double> test_input = {0.5, 0.4, 0.7};
    std::vector<double> test_labels = {0.2, 0.2, 0.2};
    double eta = 0.3;

    int epochs = 100;
    int i;
    for(i = 0; i < epochs; i++)
    {
        auto output = Q->forward_propogate_rl(test_input);
        std::cout << "(" << i << ") OUTPUT: \n" << output << std::endl;

        Q->back_propogate_rl(output, test_labels);
        Q->update_weights_rl(0.3);
    }
}