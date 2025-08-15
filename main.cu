#include <chrono>
#include <iostream>
#include <random>

#include "bitonic.cuh"

bool checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "Error encountered: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return true;
}

// runs bitonic sort, measures its performance and compares it to std::sort
void run_sort(std::vector<int> to_sort, const cudaDeviceProp &device_prop) {
  const size_t data_size = to_sort.size();

  int *device_data = nullptr;
  if (!checkCudaError(cudaMalloc(&device_data, data_size * sizeof(int)))) {
    return;
  }
  if (!checkCudaError(cudaMemcpy(device_data, to_sort.data(), data_size * sizeof(int), cudaMemcpyHostToDevice))) {
    return;
  }

  const auto cuda_start = std::chrono::high_resolution_clock::now();
  bitonic::sort(device_data, data_size, device_prop);

  cudaDeviceSynchronize();  // wait for all CUDA operations to finish
  const auto cuda_end = std::chrono::high_resolution_clock::now();

  auto sort_result = std::vector<int>(data_size, 0);
  cudaMemcpy(sort_result.data(), device_data, data_size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(device_data);

  // running std::sort for comparison
  const auto cpu_start = std::chrono::high_resolution_clock::now();
  std::sort(to_sort.begin(), to_sort.end());
  const auto cpu_end = std::chrono::high_resolution_clock::now();

  // the results of running both algorithms must be the same
  if (sort_result != to_sort) {
    std::cerr << "Results are different" << std::endl;
    return;
  }

  const auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start).count();
  const auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
  std::cout << std::format("{:8} kB {:8} ms {:8} ms", data_size / 256U, cuda_time, cpu_time) << std::endl;
}

int main() {
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

  static constexpr int min_val = std::numeric_limits<int>::min();
  static constexpr int max_val = std::numeric_limits<int>::max();
  std::random_device rd;
  std::mt19937 gen{rd()};
  auto distr = std::uniform_int_distribution<int>(min_val, max_val);

  // printing header for the performance table
  std::cout << std::format("{:10} | {:10} | {:10}", "Data size", "GPU time", "CPU time") << std::endl;

  for (size_t i = 0U, data_size = 256U; i < 19U; ++i, data_size <<= 1U) {
    std::vector<int> sample_data{};
    sample_data.reserve(data_size);
    for (size_t j = 0; j < data_size; ++j) {
      sample_data.push_back(distr(gen));
    }

    run_sort(std::move(sample_data), device_prop);
  }

  return 0;
}
