#include <iostream>
#include <random>

__device__ void swap(int& a, int& b) noexcept {
  const int temp = a;
  a = b;
  b = temp;
}

__global__ void bitonic(int* data, unsigned size) {
  extern __shared__ int loc_data[];
  const unsigned thr_id = threadIdx.x;
  const unsigned grid_id = blockIdx.x * blockDim.x + thr_id;

  loc_data[thr_id] = data[grid_id];
  __syncthreads();

  // in the outer loop we are selecting the size of the sorting region
  for (unsigned k = 2U; k <= size; k <<= 1U) {
    // in the inner loop we swap elements with their partners
    for (unsigned j = k >> 1U; j > 0U; j >>= 1U) {
      const unsigned partner_idx = thr_id ^ j;
      const bool is_left = (thr_id & k) == 0;
      const bool l2r = partner_idx > thr_id;
      const auto diff = loc_data[thr_id] - loc_data[partner_idx];
      const bool to_swap =
          l2r && ((is_left && diff > 0) || (!is_left && diff < 0));
      if (to_swap) {
        swap(loc_data[thr_id], loc_data[partner_idx]);
      }
      __syncthreads();
    }
  }

  data[grid_id] = loc_data[thr_id];
}

bool checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "Error encountered: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return true;
}

void print_array(const std::vector<int>& arr, size_t n_to_print = 32U) {
  n_to_print = std::min(n_to_print, arr.size());
  for (size_t i = 0; i < n_to_print; ++i) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
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

  cudaDeviceProp device_prop{};
  if (!checkCudaError(cudaGetDeviceProperties(&device_prop, 0))) {
    return 1;
  }

  std::cout << device_prop.name << std::endl;
  std::cout << "Device capability: " << device_prop.major << "."
            << device_prop.minor << std::endl;

  static constexpr int min_val = -10;
  static constexpr int max_val = 10;
  std::random_device rd;
  std::mt19937 gen{rd()};
  auto distr = std::uniform_int_distribution<int>(min_val, max_val);

  static constexpr size_t data_size = 32;
  std::vector<int> sample_data{};
  sample_data.reserve(data_size);
  for (size_t i = 0; i < data_size; ++i) {
    sample_data.push_back(distr(gen));
  }
  std::cout << "Initial state:" << std::endl;
  print_array(sample_data, data_size);

  const int n_threads = std::min(static_cast<int>(data_size), device_prop.maxThreadsPerBlock);
  int* device_data = nullptr;
  if (!checkCudaError(cudaMalloc(&device_data, data_size * sizeof(int)))) {
    return 1;
  }
  if (!checkCudaError(cudaMemcpy(device_data, sample_data.data(), data_size * sizeof(int), cudaMemcpyHostToDevice))) {
    return 1;
  }
  bitonic<<<1, n_threads, n_threads * sizeof(int)>>>(device_data, data_size);

  cudaMemcpy(sample_data.data(), device_data, data_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(device_data);

  std::cout << "After sorting:" << std::endl;
  print_array(sample_data, data_size);

  return 0;
}
