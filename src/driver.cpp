#include "driver.h"


int main()
{
    auto print_vec = []<typename T>(std::vector<T>& vec)
    {
        for(T const & v : vec)
            std::cout << v << "\n";
        std::cout << std::endl;
    };

    std::vector<std::string> optimisations;
    std::vector<std::string> benchmarks;

    read_file_to_vec(optimisations, DEFAULT_OPTIMISATIONS_LIST_LOCATION);
    read_file_to_vec(benchmarks, DEFAULT_BENCHMARKS_LIST_LOCATION);

    std::string test_benchmark("correlation");

    std::vector<std::string> optimisation_set(optimisations.begin() + 3, optimisations.begin() + 9);
    auto output = iterative_optimiser(test_benchmark, optimisation_set, benchmarks);

    print_vec(output);

    return 0;
}