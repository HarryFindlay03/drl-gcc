#ifndef ML_ANN_H
#define ML_ANN_H

class ML_ANN
{
    std::vector<Layer*> layers;
    size_t num_layers;
    size_t minibatch_size;

public:
    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ML_ANN(const std::vector<size_t>& layer_config);

    ~ML_ANN();

    static Eigen::MatrixXd elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs);

    Eigen::MatrixXd forward_propogate_rl(const std::vector<double>& data);

    void back_propogate_rl(const double output, const double target);
    void back_propogate_rl(const Eigen::MatrixXd& output, const std::vector<double>& targets);

    void update_weights_rl(const double eta);
};

#endif