# drl-gcc

A novel reinforcement learning framework to determine a near-optimal subset of optimisation options given a set of options and a program.

Program analysis to construct a program's feature vector is performed at the IR stage of the GCC compilation process by parsing the resultant GIMPLE statements.

### Installation Prequisites
- polybench-c-3.2 and place in root of this project
- Eigen 3.4.0 and place in /include/mlp-nn

There is an installation script `install.sh` that will create all the required folders for execution and ensure that the Eigen library has been installed in the correct place.


### Examples
Examples can be found in src/examples folder with rules already written in the `Makefile` distributed with this repository.