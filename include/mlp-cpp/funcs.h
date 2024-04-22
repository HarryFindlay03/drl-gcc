
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

/**
 * @brief Loss function of standard deep q-learning.
 * 
 * @param output 
 * @param target 
 * @param action_pos 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd dql_square_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);

/**
 * @brief loss function of standard deep q learning with error clipping between -1 and 1. This helps improve algorithm stability.
 * 
 * @param output 
 * @param target 
 * @param action_pos 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd dql_square_loss_with_error_clipping(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);

/**
 * @brief huber loss function implementation for increased stability of deep q learning algorithm
 * src: https://en.wikipedia.org/wiki/Huber_loss
 * 
 * @param output 
 * @param target 
 * @param action_pos 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd huber_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);

Eigen::MatrixXd standard_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);


#endif /* FUNCS_H */