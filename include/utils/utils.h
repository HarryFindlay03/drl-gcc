/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 14/04/2024
 * FILE LAST UPDATED: 10/05/2024
 * 
 * REQUIREMENTS: PolyBench
 * REFERENCES:
 * 
 * DESCRIPTION: Header file for utilities for drl framework.
*/


#ifndef UTILS_H
#define UTILS_H

#define DEFAULT_OPTIMISATIONS_LIST_LOCATION "data/optimisations.txt"
#define DEFAULT_BENCHMARKS_LIST_LOCATION "data/benchmark_list.txt"

#define DEFAULT_EXEC_OUTPUT_LOCATION "bin/tmp/"
#define DEFAULT_DATA_OUTPUT_LOCATION "data/tmp/tmpXX"
#define DEFAULT_PLUGIN_OUTPUT_LOCATION "data/tmp/statetmpXX.txt"

#define DEFAULT_PLUGIN_INFO "-fplugin=./statetool.dylib -fplugin-arg-statetool.dylib-filename=" DEFAULT_PLUGIN_OUTPUT_LOCATION

#define POLY_COMPILER "gcc"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>


/**
 * @brief PolyString used to help build a full optimisation string - ease of use
 */
struct PolyString
{
    /* header string contains host compiler and include directories */
    std::string header;

    /* GCC plugin string information */
    std::string plugin_info;

    /* output string contains output information for gcc */
    std::string output;

    /* contains information about the optimisations being applied */
    // std::string optimisations;
    std::vector<std::string> optimisations;


    PolyString(const std::string& header, const std::string& plugin_info, const std::string& output, const std::string& baseline)
    : header(header), plugin_info(plugin_info), output(output)
    { optimisations.push_back(baseline); };

    void reset_PolyString_optimisations() { optimisations.clear(); };

    /**
     * @brief Get the full PolyString as one string seperated by spaces in order: header - plugin_info - output - optimisations
     * 
     * @return std::string 
     */
    inline std::string get_full_PolyString()
    {
        std::string res = header + " " + plugin_info + " " + output + " ";

        for(auto const& s : optimisations)
            res += (s + " ");

        return res;
    };

    inline std::string get_no_plugin_PolyString()
    {
        std::string res = header + " " + output + " ";

        for(auto const& s : optimisations)
            res += (s + " ");

        return res;
    }

    inline std::string get_no_plugin_no_optimisations_PolyString()
    {
        return header + " " + output + " -O0";
    };
};

PolyString* construct_polybench_PolyString(const std::string& program_name, const std::string& baseline);

std::vector<std::string> read_file_to_vec(const std::string& filename);

std::string get_program_name(const std::string& benchmark);

std::string format_benchmark_string(const std::string& benchmark_to_fmt);

int get_benchmark_location(const std::string& program_name);

double run_given_string(const std::string& compile_string, const std::string& program_name);

std::vector<double> get_program_state(PolyString* ps, int num_features);

std::vector<double> read_state_vector(const std::string& filename, int num_features);

std::string opt_vec_to_string(const std::vector<std::string>& opts);

bool check_unop_compile(const std::string& unop, const std::string& program_name);

std::string get_benchmark_files(const std::string& program_name);

std::string construct_header(const std::string& program_name);

std::string strip_unop(const std::string& unop);

#endif