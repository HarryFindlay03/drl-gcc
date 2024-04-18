/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 14/03/2024
 * FILE LAST UPDATED: 18/04/2024
 * 
 * REQUIREMENTS: PolyBench
 * REFERENCES:
 * 
 * DESCRIPTION: Implementation file for utilities for drl agent.
*/


#include "utils/utils.h"


/* GLOBAL BENCHMARK AND OPTIMISATION VECTORS HELD IN MEMORY */
std::vector<std::string> benchmarks = read_file_to_vec(DEFAULT_BENCHMARKS_LIST_LOCATION);
std::vector<std::string> optimisations = read_file_to_vec(DEFAULT_OPTIMISATIONS_LIST_LOCATION);


PolyString* construct_polybench_PolyString(const std::string& program_name)
{
    PolyString* new_ps = new PolyString
    (
        construct_header(program_name),
        DEFAULT_PLUGIN_INFO,
        (get_benchmark_files(program_name) + "-DPOLYBENCH_TIME -o " + DEFAULT_EXEC_OUTPUT_LOCATION + program_name)
    );

    return new_ps;
}


std::vector<std::string> read_file_to_vec(const std::string& filename)
{
    std::vector<std::string> ret;
    std::string line;
    std::ifstream f(filename);

    if(f.is_open())
    {
        while(getline(f, line))
            ret.push_back(line);
    }
    else
    {
        std::cout << "ERROR: FILE DOES NOT EXIST!" << std::endl;
        std::exit(-1);
    }

    f.close();

    return ret;
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


std::string format_benchmark_string(const std::string& benchmark_to_fmt)
{
    int i = get_benchmark_location(benchmark_to_fmt);

    /* forming the benchmark string */
    std::string benchmark_string(POLY_COMPILER + (std::string)" -I polybench-c-3.2/utilities");

    int j;
    for(j = benchmarks.size(); j >= 0; j--)
        if(benchmarks[i][j] == '/')
            break;

    benchmark_string.append(" -I " + benchmarks[i].substr(0, j) + " polybench-c-3.2/utilities/polybench.c ");
    benchmark_string.append(benchmarks[i] + " -DPOLYBENCH_TIME -o " + DEFAULT_EXEC_OUTPUT_LOCATION + benchmark_to_fmt);

    return benchmark_string;
}


int get_benchmark_location(const std::string& program_name)
{
    int n = benchmarks.size();
    if (n == 0)
    {
        std::cout << "ERROR: benchmarks file has not been previously read!" << std::endl;
        std::exit(-1);
    }

    int i, j;
    for (i = 0; i < n; i++)
    {
        if (get_program_name(benchmarks[i]) == program_name)
            break;
    }

    // benchmark_to_fmt not found
    if (i == n)
    {
        std::cout << "ERROR: program to benchmark not found in available programs" << std::endl;
        std::exit(-1);
    }

    return i;
}


double run_given_string(const std::string& compile_string, const std::string& program_name)
{
    // ensuring folders have been created
    std::system("mkdir -p bin/tmp");
    std::system("mkdir -p data/tmp");

    // compiling the program
    std::cout << "run_given_string() compile string: " << compile_string << '\n';
    std::system(compile_string.c_str());


    // running the program
    std::string exec_string((std::string)"./" + DEFAULT_EXEC_OUTPUT_LOCATION + program_name + (std::string)" > " + DEFAULT_DATA_OUTPUT_LOCATION);
    std::cout << "Running program string: " << exec_string << '\n';
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
    std::system(((std::string) "rm -f " + DEFAULT_EXEC_OUTPUT_LOCATION + program_name).c_str());

    // delete tmp data
    std::system(((std::string)"rm " + DEFAULT_DATA_OUTPUT_LOCATION).c_str());

    return res;    
}


std::vector<double> get_program_state(PolyString* ps, int num_features)
{
    /* use stateplugin to gather program state from filename passed as plugin argument */

    /* ensure stateplugin has been built prior to running this function */

    // creating temp folder
    std::system("mkdir -p data/tmp");

    std::string exec_string = ps->get_full_PolyString();

    // read state vector
    std::system(exec_string.c_str());
    std::vector<double> prog_state = read_state_vector(DEFAULT_PLUGIN_OUTPUT_LOCATION, num_features);

    // remove tmp data
    std::system(((std::string)"rm " + DEFAULT_PLUGIN_OUTPUT_LOCATION).c_str());

    return prog_state;
}


std::vector<double> read_state_vector(const std::string& filename, int num_features)
{
    std::vector<double> res(num_features);
    std::ifstream state_file(filename.c_str());

    if(state_file.is_open())
    {
        int pos = 0;
        std::string line;
        while(getline(state_file, line))
            res[pos++] = std::stod(line);     
    }
    else
        return {-1};

    state_file.close();
    return res;
}


std::string opt_vec_to_string(const std::vector<std::string>& opts)
{
    std::string res;

    for(const auto & opt : opts)
        res += opt;

    return res;
}


bool check_unop_compile(const std::string& unop, const std::string& program_name)
{
    // compile the program to location
    std::system("mkdir -p bin/tmp");

    std::system(unop.c_str());

    const std::filesystem::path unop_path{DEFAULT_EXEC_OUTPUT_LOCATION + program_name};
    bool res = std::filesystem::exists(unop_path);

    // removing temp program
    std::system(((std::string)"rm " + DEFAULT_EXEC_OUTPUT_LOCATION + program_name).c_str());

    return res;
}


std::string construct_header(const std::string& program_name)
{
    int i = get_benchmark_location(program_name);

    /* forming the header string*/
    std::string header_string(POLY_COMPILER + (std::string) " -I polybench-c-3.2/utilities");

    int j;
    for (j = benchmarks.size(); j >= 0; j--)
        if (benchmarks[i][j] == '/')
            break;

    header_string.append(" -I " + benchmarks[i].substr(0, j));

    return header_string;
}


std::string get_benchmark_files(const std::string& program_name)
{
    int pos = get_benchmark_location(program_name);
    return ("polybench-c-3.2/utilities/polybench.c " + benchmarks[pos] + " "); 
}


std::string strip_unop(const std::string& unop)
{
    std::string res;

    res = unop.substr(0, unop.find("-O0"));

    return res;
}