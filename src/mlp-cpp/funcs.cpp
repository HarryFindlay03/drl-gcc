
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


Eigen::MatrixXd dql_square_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    return (output - target).array().square().matrix();
}


Eigen::MatrixXd standard_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    return output - target;
}