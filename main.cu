#include <iostream>
#include <random>

__device__ void swap(int &a, int &b) noexcept {
  const int tmp = a;
  a = b;
  b = tmp;
}

__global__ void bitonic(int *data) {
  extern __shared__ int loc_data[];
  const unsigned thr_id = threadIdx.x;
  const unsigned grid_id = blockIdx.x * blockDim.x + thr_id;

  loc_data[thr_id] = data[grid_id];
  __syncthreads();

  // in the outer loop we are selecting the size of the sorting region
  for (unsigned k = 2U; k <= blockDim.x; k <<= 1U) {
    // in the inner loop we swap elements with their partners
    for (unsigned j = k >> 1U; j > 0U; j >>= 1U) {
      const unsigned partner_idx =
          thr_id | j; // always greater or equal to thr_id
      const bool is_left = (grid_id & k) == 0;
      const bool to_swap =
          (is_left && (loc_data[thr_id] > loc_data[partner_idx])) ||
          (!is_left && (loc_data[thr_id] < loc_data[partner_idx]));
      if (to_swap) {
        swap(loc_data[thr_id], loc_data[partner_idx]);
      }
      __syncthreads();
    }
  }

  data[grid_id] = loc_data[thr_id];
}

__global__ void bitonic_iteration(int *data, unsigned tonic_size,
                                  unsigned span) {
  const unsigned grid_id = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned partner_idx = grid_id | span;
  const bool is_left = (grid_id & tonic_size) == 0;
  const bool to_swap = (is_left && (data[grid_id] > data[partner_idx])) ||
          (!is_left && (data[grid_id] < data[partner_idx]));
  if (to_swap) {
    swap(data[grid_id], data[partner_idx]);
  }
}

__host__ void bitonic_global(int *data, unsigned data_size, unsigned grid_size,
                             unsigned block_size) {
  unsigned size = block_size << 1U;
  while (size <= data_size) {
    for (unsigned span = size >> 1U; span > 0U; span >>= 1U) {
      bitonic_iteration<<<grid_size, block_size>>>(data, size, span);
    }
    size <<= 1U;
  }
}

bool checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "Error encountered: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return true;
}

void print_array(const std::vector<int> &arr, size_t n_to_print = 32U) {
  n_to_print = std::min(n_to_print, arr.size());
  for (size_t i = 0; i < n_to_print; ++i) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

// rounds value down to the nearest power of 2 such that the result
// is greater or equal to the input value
// returns 1 if the input is less than 2
unsigned ceil_2(unsigned val) {
  if (val <= 1U) {
    return 1U;
  }

  unsigned res = 2U;
  while (res < val) {
    res <<= 1U;
  }

  return res;
}

// rounds value down to the nearest power of 2 such that the result
// is less or equal to the input value. Never returns zero
unsigned floor_2(unsigned val) {
  const unsigned res = ceil_2(val);

  if (res < 2U) {
    return res;
  }
  return res == val ? res : res >> 1U;
}

unsigned get_block_size(unsigned data_size, unsigned max_threads_per_block,
                        unsigned shared_mem_per_block) {
  // we assume that data_size is already a power of two
  const unsigned limit = std::min(floor_2(max_threads_per_block),
                                  floor_2(shared_mem_per_block / sizeof(int)));
  return std::min(limit, data_size);
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
  std::cout << "Device capability: " << device_prop.major << "."
            << device_prop.minor << std::endl;

  static constexpr int min_val = -10;
  static constexpr int max_val = 10;
  std::random_device rd;
  std::mt19937 gen{rd()};
  auto distr = std::uniform_int_distribution<int>(min_val, max_val);

  const size_t data_size = 2048U;

  std::vector<int> sample_data{};
  sample_data.reserve(data_size);
  for (size_t i = 0; i < data_size; ++i) {
    sample_data.push_back(distr(gen));
  }
  std::cout << "Initial state:" << std::endl;
  print_array(sample_data, data_size);

  int *device_data = nullptr;
  if (!checkCudaError(cudaMalloc(&device_data, data_size * sizeof(int)))) {
    return 1;
  }
  if (!checkCudaError(cudaMemcpy(device_data, sample_data.data(),
                                 data_size * sizeof(int),
                                 cudaMemcpyHostToDevice))) {
    return 1;
  }

  // finding the block size
  const unsigned block_size = get_block_size(
      data_size, device_prop.maxThreadsPerBlock, device_prop.sharedMemPerBlock);
  const unsigned grid_size = data_size / block_size;

  bitonic<<<grid_size, block_size, block_size * sizeof(int)>>>(device_data);
  if (grid_size > 1U) {
    bitonic_global(device_data, data_size, grid_size, block_size);
  }

  cudaMemcpy(sample_data.data(), device_data, data_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  cudaFree(device_data);

  std::cout << "After sorting:" << std::endl;
  print_array(sample_data, data_size);

  return 0;
}
