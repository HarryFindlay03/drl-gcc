
#include "mlp-cpp/funcs.h"

Eigen::MatrixXd mlp_sigmoid(const Eigen::MatrixXd& input, bool deriv)
{
    Eigen::MatrixXd res(input.rows(), input.cols());
    int i, j;

    if(deriv)
    {
        Eigen::MatrixXd unprime_sig = mlp_sigmoid(input, false);

        for(i = 0; i < res.rows(); i++)
        {
            for(j = 0; j < res.cols(); j++)
            {
                res(i, j) = unprime_sig(i, j) * (1 - unprime_sig(i, j));
            }
        }

        return res;
    }

    for(i = 0; i < res.rows(); i++)
    {
        for(j = 0; j < res.cols(); j++)
        {
            res(i, j) = 1 / (1 + exp(-(input(i, j))));
        }
    }

    return res;
}


// Eigen::MatrixXd mlp_tanh(const Eigen::MatrixXd& input, bool deriv)
// {

// }


Eigen::MatrixXd mlp_ReLU(const Eigen::MatrixXd& input, bool deriv)
{
    Eigen::MatrixXd res(input.rows(), input.cols());

    int i, j;
    if(deriv)
    {
        for(i = 0; i < res.rows(); i++)
        {
            for(j = 0; j < res.cols(); j++)
            {
                if(input(i, j) <= 0)
                    res(i, j) = 0;
                else
                    res(i, j) = 1;
            }
        }

        return res;
    }

    for(i = 0; i < res.rows(); i++)
    {
        for(j = 0; j < res.cols(); j++)
        {
            if(input(i, j) <= 0)
                res(i, j) = 0;
            else
                res(i, j) = input(i, j);
        }
    }

    return res;
}


Eigen::MatrixXd mlp_linear(const Eigen::MatrixXd& input, bool deriv)
{
    // derivative of linear activation function is always 1
    if(deriv)
        return Eigen::MatrixXd::Ones(input.rows(), input.cols());

    Eigen::MatrixXd res = input;
    return res;
}


/* INITIALISER FUNCTIONS */


void xiaver_initialiser(Eigen::MatrixXd& mat, int fan_in, int fan_out, rand_helper* rnd)
{
    int i, j;
    for(i = 0; i < mat.rows(); i++)
    {
        for(j = 0; j < mat.cols(); j++)
        {
            double rhs = std::sqrt(6.0 / (fan_in + fan_out));
            double lhs = -(rhs);
            mat(i, j) = rnd->random_double_range(lhs, rhs);
        }
    }
}


void he_normal_initialiser(Eigen::MatrixXd& mat, int fan_in, int fan_out, rand_helper* rnd)
{
    int i, j;
    for(i = 0; i < mat.rows(); i++)
    {
        for(j = 0; j < mat.cols(); j++)
        {
            double sd = std::sqrt(2.0 / fan_in);
            mat(i, j) = rnd->normal_distribution(0.0, sd);
        }
    }
}


/* LOSS FUNCTIONS */


Eigen::MatrixXd dql_square_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos)
{
    // update only at the action pos position
    // this results in a gradient update only for action j at node j
    Eigen::MatrixXd new_target = Eigen::MatrixXd::Zero(target.rows(), target.cols());
    new_target(0, action_pos) = target(0, action_pos);

    return (output - new_target).array().square().matrix();
}


Eigen::MatrixXd dql_square_loss_with_error_clipping(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos)
{
    double err_clip_val = 0.5; // keep positive

    Eigen::MatrixXd new_target = Eigen::MatrixXd::Zero(target.rows(), target.cols());
    new_target(0, action_pos) = target(0, action_pos);

    Eigen::MatrixXd res = (output - new_target);

    // error clipping in interval [-(err_clip_val), err_clip_val].
    if(res(0, action_pos) > err_clip_val)
        res(0, action_pos) = err_clip_val;
    else if(res(0, action_pos) < (-1 * err_clip_val))
        res(0, action_pos) = (-1 * err_clip_val);

    return res.array().square().matrix().eval();
}


Eigen::MatrixXd huber_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos)
{
    double huber_delta = 1; // clipping between -1 and 1

    Eigen::MatrixXd new_target = Eigen::MatrixXd::Zero(target.rows(), target.cols());
    new_target(0, action_pos) = target(0, action_pos);

    bool status = (std::fabs(output(0, action_pos) - new_target(0, action_pos)) <= huber_delta);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(target.rows(), target.cols());
    if(status)
        res(0, action_pos) = 0.5 * (std::pow((output(0, action_pos) - new_target(0, action_pos)), 2));
    else
        res(0, action_pos) = huber_delta * ((std::fabs(output(0, action_pos) - new_target(0, action_pos))) - (0.5 * huber_delta));

    return res;
}


Eigen::MatrixXd standard_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos)
{
    return output - target;
}


/* SCALING FUNCTIONS */


std::vector<double> vec_min_max_scaling(const std::vector<double>& in)
{
    std::vector<double> res(in.size());

    auto find_val = [](const std::vector<double>& vec_in, bool max)
    {
        int val_pos = 0;
        int x;
        if(max)
        {
            for(x = 1; x < vec_in.size(); x++)
                if(vec_in[x] > vec_in[val_pos])
                    val_pos = x;
        }
        else
        {
            for(x = 1; x < vec_in.size(); x++)
                if(vec_in[x] < vec_in[val_pos])
                    val_pos = x;
        }

        return val_pos;
    };

    double min = in[find_val(in, false)];
    double max = in[find_val(in, true)];

    int i;
    for(i = 0; i < res.size(); i++)
        res[i] = (in[i] - min) / (max - min);

    return res;
}