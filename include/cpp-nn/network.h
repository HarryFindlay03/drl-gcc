#include <iostream>
#include <random>
#include <ctime>

#include "Eigen/Dense"

#include "Layer.h"
#include "ML_ANN.h"

/* GLOBAL ACTIVATION FUNCTIONS */

Eigen::MatrixXd vector_f_sigmoid_rl(const Eigen::MatrixXd& in, bool deriv);