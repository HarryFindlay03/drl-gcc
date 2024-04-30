# drl-gcc

A novel reinforcement learning framework to determine a near-optimal subset of optimisation options given a set of options and a program.

Program analysis to construct a program's feature vector is performed at the IR stage of the GCC compilation process by parsing the resultant GIMPLE statements.

### Installation Prequisites
- polybench-c-3.2 and place in root of this project
- Eigen 3.4.0 and place in /include/mlp-nn

There is an installation script `install.sh` that will create all the required folders for execution and ensure that the Eigen library has been installed in the correct place.


### Examples
Examples can be found in src/examples folder with rules already written in the `Makefile` distributed with this repository.

A basic example to train an agent on a provided action set and training program space is given below:

```C++
#include "dqn/Agent.h"

#define MY_SEED 321

int main()
{
    // input layer - num features parsed in statetool
    // output layer - size of action space

    std::vector<std::string> actions = read_file_to_vec("data/action_spaces/LOOPS_CSE_actionspace.txt");

    // to include NOP operation
    actions.push_back(NOP);

    int output_layer_size = actions.size();

    std::vector<std::string> training_programs = read_file_to_vec("data/program_spaces/training_programs_loops_cse.txt");


    std::vector<int> network_config = {7, 30, 30, 30, output_layer_size};

    unsigned int buffer_size = 300;
    unsigned int copy_period = 4;
    unsigned int number_episodes = 100;
    unsigned int episode_length = 7;
    double discount_rate = 0.9;
    double learning_rate = 0.001;

    rand_helper* rnd = new rand_helper(MY_SEED);

    Agent* ag = new Agent(network_config, actions, training_programs, buffer_size, copy_period, number_episodes, episode_length, discount_rate, learning_rate, rnd, true);

    double epsilon = 0.3;
    ag->train_optimiser(epsilon);

    return 0;
}
```


### Code Documentation
If functions are not self explanatory docstrings have been provided in the relevant header (.h) files.