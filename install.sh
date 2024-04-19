# Making folders for building and linking steps
mkdir -p "bin"
echo "Creating bin folder."
mkdir -p "build"
echo "Creating build folder."

# Making temp folders for data storage during optimiser execution
mkdir -p "data/tmp"
mkdir -p "bin/tmp"
echo "Creating temp folders for data storage."

# Checking that folders have been created correctly
echo "Checking folder structure existence."

structure_correct=1

if [ ! -d "bin" ]; then
    echo "Linking bin folder NOT created."
    structure_correct=0
fi

if [ ! -d "build" ]; then
    echo "Building build folder NOT created."
    structure_correct=0
fi

if [ ! -d "data/tmp" ]; then
    echo "Temp data folder NOT created, program will not run correctly!"
    structure_correct=0
fi

if [ ! -d "bin/tmp" ]; then
    echo "Temp exec folder NOT created, program will not run correctly!"
    structure_correct=0
fi

# Checking that the Eigen library has been installed correctly
echo "Checking Eigen library has been installed correctly."
if [ ! -d "include/mlp-cpp/Eigen" ]; then
    echo "Eigen library not installed, program will not work!"
    echo "Please copy the Eigen header library into include/mlp-cpp/."
    structure_correct=0
fi

if [ $structure_correct -eq 1 ]; then
    echo "Folder structure correct, program expected to run okay."
fi

if [ $structure_correct -eq 0 ]; then
    echo "Folder structure INcorrect, please remedy above error messages to fix problems and rerun this script."
fi