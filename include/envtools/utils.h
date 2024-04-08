#ifndef UTILS_H
#define UTILS_H

#define DEFAULT_OPTIMISATIONS_LIST_LOCATION "data/optimisations.txt"
#define DEFAULT_BENCHMARKS_LIST_LOCATION "data/benchmark_list.txt"

#define DEFAULT_EXEC_OUTPUT_LOCATION "bin/tmp/"
#define DEFAULT_DATA_OUTPUT_LOCATION "data/tmp/tmpXX"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>


void read_file_to_vec(std::vector<std::string>& vec, const std::string& filename);

std::string get_program_name(const std::string& benchmark);

std::string format_benchmark_string(const std::string& benchmark_to_fmt, const std::vector<std::string>& benchmarks);

double run_given_string(const std::string& compile_string, const std::string& program_name);

std::vector<double> get_program_state_profile(const std::string& compile_string);

std::vector<double> get_program_state(const std::string& compile_string);

std::string opt_vec_to_string(const std::vector<std::string>& opts);

#endif