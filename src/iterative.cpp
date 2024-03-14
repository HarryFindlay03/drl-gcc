#include "iterative.h"


std::vector<double> iterative_optimiser(const std::string& benchmark, const std::vector<std::string>& optimisation_set, const std::vector<std::string>& benchmarks)
{
    // format benchmark string

    // for each possible optimisation set in optimisation_set
        // apply the optimisation and run the program storing the execution time

    std::vector<double> perf_vec(optimisation_set.size());

    auto vec_to_string = [](const std::vector<std::string>& vec)
    {
        std::string res;
        for(auto const & v : vec)
            res += (" " + v);

        return res;
    };

    int i;
    for(i = 0; i < optimisation_set.size(); i++)
    {
        std::vector<std::string> slice(optimisation_set.begin(), optimisation_set.begin() + i + 1);
        std::string benchmark_string = format_benchmark_string(benchmark, benchmarks) + vec_to_string(slice);

        perf_vec[i] = run_given_string(benchmark_string, benchmark);
    }

    return perf_vec;
}