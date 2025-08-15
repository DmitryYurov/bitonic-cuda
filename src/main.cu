#include <chrono>
#include <iostream>
#include <random>

#include "bitonic.cuh"

/**
 * @brief Checks CUDA runtime API errors and outputs an error message if any
 *
 * This function examines a CUDA runtime API error code and outputs an error message
 * to the standard error stream if the error code indicates a failure.
 *
 * @param err CUDA runtime API error code to check
 * @return true if no error occurred (cudaSuccess), false otherwise
 */
bool checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "Error encountered: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return true;
}

template <typename Deleter> using cuda_ptr = std::unique_ptr<int[], Deleter>; /// A smart pointer to handle cuda-side arrays

/**
 * @brief Allocates an array of integers on the device side and returns cuda_ptr to it.
 *
 * @param size The size of the integer array to allocate
 * @return a smart pointer to the allocated array, empty in case of allocation failure
 */
auto allocate_device_memory(size_t size) {
  auto deleter = [](int *ptr) {
    if (ptr != nullptr)
      cudaFree(ptr);
  };

  auto result = cuda_ptr<decltype(deleter)>(nullptr, std::move(deleter));
  int *device_data = nullptr;
  if (checkCudaError(cudaMalloc(&device_data, size * sizeof(int)))) {
    result.reset(device_data);
  }

  return result;
}

/**
 * @brief Calculates the execution time between two time points in milliseconds
 *
 * @param start The starting time point of the measurement interval
 * @param end The ending time point of the measurement interval
 * @return The duration between start and end in milliseconds as a double value
 */
double exec_time(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
  using namespace std::chrono;
  const auto time_us = duration_cast<microseconds>(end - start).count();
  return static_cast<double>(time_us) / 1000.0;
}

/**
 * @brief Runs bitonic sort on GPU, measures its performance and compares it to std::sort
 *
 * This function performs the following steps:
 * 1. Allocates device memory and copies input data to GPU
 * 2. Runs bitonic sort on GPU and measures execution time
 * 3. Copies results back to host memory
 * 4. Runs std::sort on CPU for comparison and measures execution time
 * 5. Validates that both sorting results match
 * 6. Prints performance metrics
 *
 * @param to_sort Vector containing integers to sort
 * @param device_prop CUDA device properties used for sorting
 */
void run_sort(std::vector<int> to_sort, const cudaDeviceProp &device_prop) {
  const size_t data_size = to_sort.size();

  auto device_data = allocate_device_memory(data_size);
  if (device_data == nullptr) {
    std::cerr << "Failed to allocate device memory" << std::endl;
    return;
  }

  if (!checkCudaError(cudaMemcpy(device_data.get(), to_sort.data(), data_size * sizeof(int), cudaMemcpyHostToDevice))) {
    return;
  }

  const auto cuda_start = std::chrono::high_resolution_clock::now();
  bitonic::sort(device_data.get(), data_size, device_prop);

  cudaDeviceSynchronize(); // wait for all CUDA operations to finish
  const auto cuda_end = std::chrono::high_resolution_clock::now();

  auto sort_result = std::vector<int>(data_size, 0);
  cudaMemcpy(sort_result.data(), device_data.get(), data_size * sizeof(int), cudaMemcpyDeviceToHost);

  // running std::sort for comparison
  const auto cpu_start = std::chrono::high_resolution_clock::now();
  std::sort(to_sort.begin(), to_sort.end());
  const auto cpu_end = std::chrono::high_resolution_clock::now();

  // the results of running both algorithms must be the same
  if (sort_result != to_sort) {
    std::cerr << "Results are different" << std::endl;
    return;
  }

  const auto cuda_time = exec_time(cuda_start, cuda_end);
  const auto cpu_time = exec_time(cpu_start, cpu_end);
  std::cout << std::format("{:8} kB {:8.2f} ms {:8.2f} ms", data_size / 256U, cuda_time, cpu_time) << std::endl;
}

/**
 * @brief Generates a vector of random integers
 *
 * This function creates a random number generator (RNG) with a random seed on the first call
 * and reuses it for subsequent calls. It generates random integers uniformly distributed
 * between minimum and maximum possible integer values.
 *
 * @param data_size Number of random integers to generate
 * @return std::vector<int> Vector containing the generated random integers
 */
std::vector<int> generate_random_data(unsigned data_size) {
  static const auto init_rng = []() {
    std::random_device rd;
    return std::mt19937{rd()};
  };
  static auto rng = init_rng();

  static constexpr int min_val = std::numeric_limits<int>::min();
  static constexpr int max_val = std::numeric_limits<int>::max();
  auto distr = std::uniform_int_distribution<int>(min_val, max_val);

  std::vector<int> sample_data{};
  sample_data.reserve(data_size);
  for (size_t j = 0; j < data_size; ++j) {
    sample_data.push_back(distr(rng));
  }

  return sample_data;
}

/**
 * @brief Parses a string argument into an unsigned integer value, ensuring it's a power of 2
 *
 * This function attempts to convert a string argument into an unsigned integer.
 * If the conversion fails or the value is too large, it returns std::nullopt.
 * The function also ensures the resulting value is a power of 2 by rounding down
 * to the nearest power of 2 if necessary.
 *
 * @param arg String argument to parse
 * @return std::optional<unsigned> The parsed and validated unsigned value, or std::nullopt if parsing failed
 */
std::optional<unsigned> parse_size(const std::string &arg) {
  unsigned size = 0;
  try {
    const auto read_val = std::stoul(arg);
    if (read_val > std::numeric_limits<unsigned>::max()) {
      throw std::invalid_argument{"Size argument is too large"};
    }
    size = static_cast<unsigned>(read_val);
  } catch (const std::invalid_argument &ex) {
    std::cerr << "Invalid size argument: " << arg << "(" << ex.what() << ")" << std::endl;
    std::cerr << "Proceeding with default execution..." << std::endl;
    return std::nullopt;
  } catch (const std::exception &ex) {
    std::cerr << "Exception by parsing input arguments: " << ex.what() << std::endl;
    std::cerr << "Proceeding with default execution..." << std::endl;
    return std::nullopt;
  }

  const auto on_floor = bitonic::utilities::floor_2(size);
  if (size != on_floor) {
    std::cerr << "Input size was rounded down to " << on_floor << std::endl;
    return static_cast<unsigned>(on_floor);
  }

  return size;
}

/**
 * @brief Generates a vector of sizes for testing the bitonic sort algorithm
 *
 * This function generates test data sizes in one of two ways:
 * 1. If a user size is provided, returns a vector with just that size
 * 2. If no user size is provided, returns a vector with sizes starting from min_size (256)
 *    and doubling 19 times (up to 256 * 2^18)
 *
 * @param user_size Optional user-specified size to use
 * @return std::vector<unsigned> Vector containing the size(s) to test
 */
std::vector<unsigned> get_sizes(std::optional<unsigned> user_size) {
  static constexpr unsigned min_size = 256U;
  if (user_size.has_value()) {
    return {user_size.value()};
  }

  auto result = std::vector<unsigned>{};
  for (size_t i = 0U, data_size = min_size; i < 19U; ++i, data_size <<= 1U) {
    result.push_back(data_size);
  }

  return result;
}

/**
 * @brief Main entry point for the bitonic sort demonstration program
 *
 * This program demonstrates the bitonic sort implementation on a CUDA-capable GPU by:
 * 1. Detecting and initializing the first available CUDA device
 * 2. Running the sort on different data sizes and comparing performance with CPU sort
 * 3. Either using a user-specified size (if provided as command line argument) or
 *    running tests with sizes from 256 up to 256 * 2^18
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings. argv[1] if provided should be
 *            the desired test data size (must be a power of 2)
 * @return int 0 on successful execution, 1 on CUDA errors
 */
int main(int argc, char **argv) {
  int device_count = -1;
  if (!checkCudaError(cudaGetDeviceCount(&device_count))) {
    return 1;
  }

  if (device_count < 1) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 0;
  }

  cudaSetDevice(0); // setting the first available device

  cudaDeviceProp device_prop{};
  if (!checkCudaError(cudaGetDeviceProperties(&device_prop, 0))) {
    return 1;
  }

  std::cout << device_prop.name << std::endl;
  std::cout << "Device capability: " << device_prop.major << "." << device_prop.minor << std::endl;

  // printing header for the performance table
  std::cout << std::format("{:10} | {:10} | {:10}", "Data size", "GPU time", "CPU time") << std::endl;

  const auto user_size = argc > 1 ? parse_size(argv[1]) : std::nullopt;

  const auto data_sizes = get_sizes(user_size);
  for (const auto data_size : data_sizes) {
    run_sort(generate_random_data(data_size), device_prop);
  }

  return 0;
}
