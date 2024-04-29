/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 25/04/2024
 * 
 * REQUIREMENTS:
 * REFERENCES: 
 * 
 * DESCRIPTION: Class implementation of non-ml optimisers
*/


#include "non-ml/non-ml.h"


std::vector<std::string> random_optimiser(const std::string& program_name, const std::vector<std::string>& action_space, const std::string& baseline, int iterations, rand_helper* rnd_helper)
{
    std::vector<std::string> ret;
    double best_runtime = std::numeric_limits<double>::max();

    int i;
    for(i = 0; i < iterations; i++)
    {
        std::cout << "Iteration: " << i << '\n';
        std::vector<std::string> as_copy(action_space);

        // shuffle the copy and return the first ran_len items
        int ran_len = rnd_helper->random_int_range(0, action_space.size() - 1);

        rnd_helper->rnd_shuffle(as_copy);

        std::vector<std::string> shuffle_res(as_copy.begin(), as_copy.begin() + ran_len);
        std::string opt_string = opt_vec_to_string(shuffle_res);
        std::cout << "Optimisations chosen: " << opt_string << '\n';

        // get runtime
        double curr_runtime = run_given_string((format_benchmark_string(program_name) + " " + baseline + " " + opt_string), program_name);

        std::cout << "Best so far: ";

        bool status = (curr_runtime < best_runtime);
        if(status)
        {
            best_runtime = curr_runtime;
            ret = shuffle_res;
        }

        (status) ? std::cout << "True" : std::cout << "False";
        std::cout << "\nRuntime: " << curr_runtime << "\n\n";
    }

    return ret;
}