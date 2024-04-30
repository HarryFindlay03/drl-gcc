#include "non-ml/non-ml.h"
#include "utils/utils.h"

#define MY_SEED 321

int main()
{
    std::vector<std::string> actions = read_file_to_vec("data/action_spaces/LOOPS_CSE_actionspace.txt");

    int output_layer_size = actions.size();

    rand_helper* rnd = new rand_helper(MY_SEED);

    std::string program_name = "jacobi-2d-imper";
    std::string baseline = "-O1";

    std::cout << "RANDOM OPTIMISER\n";
    std::cout << "Progam name: " << program_name << "\n";
    std::vector<std::string> rand_opts = random_optimiser(program_name, actions, baseline, 100, rnd);
    std::cout << opt_vec_to_string(rand_opts) << "\n";


    return 0;
}