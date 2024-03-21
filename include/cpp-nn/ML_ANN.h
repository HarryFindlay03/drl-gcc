#ifndef ML_ANN_H
#define ML_ANN_H

class ML_ANN
{
    size_t num_layers;
    size_t minibatch_size;

public:
    std::vector<Layer*> layers;

    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ML_ANN(const std::vector<size_t>& layer_config);

    ~ML_ANN();

    static Eigen::MatrixXd elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs);

    Eigen::MatrixXd forward_propogate(const Eigen::MatrixXd& data);
    double forward_propogate_rl(const std::vector<double>& data);

    void back_propogate(const Eigen::MatrixXd& yhat, const Eigen::MatrixXd& labels);
    void back_propogate_rl(const double output, const double target);

    void update_weights(size_t learning_rate);
    void update_weights_rl(const double eta);

    size_t get_num_layers() { return num_layers; }
};

#endif