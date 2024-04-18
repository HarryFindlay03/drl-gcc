
#ifndef FUNCS_H
#define FUNCS_H

#include "Eigen/Core"

#include "utils/rand_helper.h"

/* ACTIVATION FUNCTIONS */
/* All activation functions work coefficient wise on the given input matrix */

Eigen::MatrixXd mlp_sigmoid(const Eigen::MatrixXd& input, bool deriv);

// Eigen::MatrixXd mlp_tanh(const Eigen::MatrixXd& input, bool deriv)
// {

// }

Eigen::MatrixXd mlp_ReLU(const Eigen::MatrixXd& input, bool deriv);

Eigen::MatrixXd mlp_linear(const Eigen::MatrixXd& input, bool deriv);

/* INITIALISER FUNCTIONS */

void xiaver_initialiser(Eigen::MatrixXd& mat, int fan_in, int fan_out, rand_helper* rnd);

void he_normal_initialiser(Eigen::MatrixXd& mat, int fan_in, int fan_out, rand_helper* rnd);

/* LOSS FUNCTIONS */

Eigen::MatrixXd dql_square_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);

Eigen::MatrixXd standard_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);

#endif /* FUNCS_H */