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

#define POLY_COMPILER "gcc"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>


void read_file_to_vec(std::vector<std::string>& vec, const std::string& filename);

std::string get_program_name(const std::string& benchmark);

std::string format_benchmark_string(const std::string& benchmark_to_fmt, const std::vector<std::string>& benchmarks);

double run_given_string(const std::string& compile_string, const std::string& program_name);

std::vector<double> get_program_state(const std::string& unop_string, const std::string& optimisations, int num_features);

std::vector<double> read_state_vector(const std::string& filename, int num_features);

std::string opt_vec_to_string(const std::vector<std::string>& opts);

bool check_unop_compile(const std::string& unop, const std::string& program_name);

std::string construct_unop(const std::string& program_name, const std::vector<std::string>& all_benchmarks);

std::string strip_unop(const std::string& unop);

#endif