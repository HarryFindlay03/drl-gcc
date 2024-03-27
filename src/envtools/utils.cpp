#include "envtools/utils.h"


void read_file_to_vec(std::vector<std::string>& vec, const std::string& filename)
{
    std::string line;
    std::ifstream f(filename);

    if(f.is_open())
    {
        while(getline(f, line))
            vec.push_back(line);
    }
    else
    {
        std::cout << "ERROR: FILE DOES NOT EXIST!" << std::endl;
        std::exit(-1);
    }

    f.close();

    return;
}


std::string get_program_name(const std::string& benchmark)
{
    int i, j;
    
    // find last '/'
    for(i = benchmark.size(); i >= 0; --i)
    {
        if(benchmark[i] == '/')
            break;
    }

    // find last '.'
    for(j = benchmark.size(); j >= 0; --j)
    {
        if(benchmark[j] == '.')
            break;
    }

    return benchmark.substr(i + 1, (j - i - 1));
}


std::string format_benchmark_string(const std::string& benchmark_to_fmt, const std::vector<std::string>& benchmarks)
{
    int n = benchmarks.size();
    if(n == 0)
    {
        std::cout << "ERROR: benchmarks file has not been previously read!" << std::endl;
        std::exit(-1);
    }

    int i, j;
    for(i = 0; i < n; i++)
    {
        if(get_program_name(benchmarks[i]) == benchmark_to_fmt)
            break;
    }

    // benchmark_to_fmt not found
    if(i == n)
    {
        std::cout << "ERROR: program to benchmark not found in available programs" << std::endl;
        std::exit(-1);
    }

    /* forming the benchmark string */
    std::string benchmark_string("gcc-13 -I polybench-c-3.2/utilities");

    for(j = n; j >= 0; j--)
        if(benchmarks[i][j] == '/')
            break;

    benchmark_string.append(" -I " + benchmarks[i].substr(0, j) + " polybench-c-3.2/utilities/polybench.c ");
    benchmark_string.append(benchmarks[i] + " -DPOLYBENCH_TIME -o " + DEFAULT_EXEC_OUTPUT_LOCATION + benchmark_to_fmt);

    return benchmark_string;
}


double run_given_string(const std::string& compile_string, const std::string& program_name)
{
    // ensuring folders have been created
    std::system("mkdir -p bin/tmp");
    std::system("mkdir -p data/tmp");

    // compiling the program
    std::system(compile_string.c_str());


    // running the program
    std::string exec_string((std::string)"./" + DEFAULT_EXEC_OUTPUT_LOCATION + program_name + (std::string)" > " + DEFAULT_DATA_OUTPUT_LOCATION);
    std::system(exec_string.c_str());

    // extracting the program execution time and cleaning up
    double res = -1;
    std::ifstream output_file(DEFAULT_DATA_OUTPUT_LOCATION);

    if(output_file.is_open())
    {
        std::string line;
        while(getline(output_file, line))
            res = std::stod(line);
    }
    else
    {
        std::cout << "ERROR: DURING PROGRAM RUNTIME EXTRACTION - CONTINUING" << std::endl;
        output_file.close();
        return -1;
    }

    output_file.close();

    // delete executable
    std::system(((std::string)"rm " + DEFAULT_EXEC_OUTPUT_LOCATION + program_name).c_str());

    // delete tmp data
    std::system(((std::string)"rm " + DEFAULT_DATA_OUTPUT_LOCATION).c_str());

    return res;    
}


std::string opt_vec_to_string(const std::vector<std::string>& opts)
{
    std::string res;

    for(const auto & opt : opts)
        res += opt;

    return res;
}