/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 14/04/2024
 * FILE LAST UPDATED: 25/05/2024
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
#include <cmath>


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

    /* baseline optimisation level */
    std::string optimisation_baseline;

    /* contains information about the optimisations being applied */
    std::vector<std::string> optimisations;

    /* stored for easy use in performance timing functions */
    std::string program_name; 

    PolyString(const std::string& program_name, const std::string& plugin_info, const std::string& output, const std::string& baseline);

    void reset_PolyString_optimisations();

    /**
     * @brief reset the PolyString environment with a new program name - clear optimisations and update header and ouptut information
     * 
     * @param new_program_name 
     */
    void reset_PolyString_environment(const std::string& new_program_name);

    /**
     * @brief Get the full PolyString as one string seperated by spaces in order: header - plugin_info - output - optimisations
     * 
     * @return std::string 
     */
    std::string get_full_PolyString();

    std::string get_no_plugin_PolyString();

    std::string get_no_plugin_no_optimisations_PolyString();
};


/* POLYBENCH HELPER FUNCTIONS */


/**
 * @brief Return a pointer to a newly constructed PolyBench environment struct given a program name and a baseline optimisation level
 * 
 * @param program_name 
 * @param baseline 
 * @return PolyString* 
 */
PolyString* construct_polybench_PolyString(const std::string& program_name, const std::string& baseline);

/**
 * @brief Returns a string constructed with required polybench information for a correct compile string, function used within construct_polybench_PolyString.
 * 
 * @param program_name 
 * @return std::string 
 */
std::string get_benchmark_files(const std::string& program_name);

/**
 * @brief Returns a string with necessary header information for a polybench compile string, used within construct_polybench_PolyString.
 * 
 * @param program_name 
 * @return std::string 
 */
std::string construct_header(const std::string& program_name);


/* FORMATTING HELPER FUNCTIONS */


/**
 * @brief Helper function to read a line seperated list of information into a string vector container.
 * 
 * @param filename 
 * @return std::vector<std::string> 
 */
std::vector<std::string> read_file_to_vec(const std::string& filename);

/**
 * @brief Helper function to parse the program name from a polybench benchmark location string.
 * 
 * @param benchmark 
 * @return std::string 
 */
std::string get_program_name(const std::string& benchmark);

/**
 * @brief Returns a formated compile string with respect to polybench header and include rules, see polybench readme.
 * 
 * @param benchmark_to_fmt 
 * @return std::string 
 */
std::string format_benchmark_string(const std::string& benchmark_to_fmt);

/**
 * @brief Returns a string joined by whitespaces of an optimisations vector often used in the PolyString struct.
 * 
 * @param opts 
 * @return std::string 
 */
std::string opt_vec_to_string(const std::vector<std::string>& opts);

/**
 * @brief Helper function used within format_benchmark_string to help construct the compile string.
 * 
 * @param program_name 
 * @return int 
 */
int get_benchmark_location(const std::string& program_name);


/* ANALYSIS FUNCTIONS */

/**
 * @brief Runs a given (polybench) compile string and returns the number of seconds that the string takes to run.
 * 
 * @param compile_string 
 * @param program_name 
 * @return double 
 */
double run_given_string(const std::string& compile_string, const std::string& program_name);

/**
 * @brief Returns a state vector of the current environment by utilising the statetool plugin.
 * 
 * @param ps 
 * @param num_features 
 * @return std::vector<double> 
 */
std::vector<double> get_program_state(PolyString* ps, int num_features);

/**
 * @brief Helper function used within get_program_state to read outputted program state from given filename passed as a plugin argument to statetool.
 * 
 * @param filename 
 * @param num_features 
 * @return std::vector<double> 
 */
std::vector<double> read_state_vector(const std::string& filename, int num_features);

/**
 * @brief Computes the relative change of the initial runtime and the new runtime as we want the sign to be taken account of.
 * 
 * @param new_runtime 
 * @param initial_runtime 
 * @return double 
 */
double relative_change_reward(const double new_runtime, const double initial_runtime);


/* TEST FUNCTIONS */


bool check_unop_compile(const std::string& unop, const std::string& program_name);


#endif