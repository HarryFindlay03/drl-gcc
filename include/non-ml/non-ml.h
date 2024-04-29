/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 25/04/2024
 * 
 * REQUIREMENTS: 
 * REFERENCES: 
 * 
 * DESCRIPTION: Class definition for non-ml optimisers
*/

#ifndef NON_ML_H
#define NON_ML_H

#include <vector>
#include <string>
#include <limits>

#include "utils/utils.h"
#include "utils/rand_helper.h"


std::vector<std::string> iterative_optimiser(const std::string& program_name, const std::vector<std::string>& action_space);

std::vector<std::string> random_optimiser(const std::string& program_name, const std::vector<std::string>& action_space, const std::string& baseline, int iterations, rand_helper* rnd_helper);


#endif