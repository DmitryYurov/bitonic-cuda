#include <iostream>

__device__ void swap(float& a, float& b) noexcept {
  const float temp = a;
  a = b;
  b = temp;
}

__global__ void bitonic(float *data, unsigned size) {
  extern __shared__ float loc_data[];
  const unsigned thr_id = threadIdx.x;
  const unsigned grid_id = blockIdx.x * blockDim.x + thr_id;

  loc_data[thr_id] = data[grid_id];
  __syncthreads();

  // in the outer loop we are selecting the size of the sorting region
  for (unsigned k = 2U; k < size; k <<= 1U) {
    // in the inner loop we swap elements with their partners
    for (unsigned j = k >> 1U; j > 0U; j >>= 1U) {
      unsigned partner_idx = thr_id ^ j;
      const bool is_left = thr_id & k == 0;
      const bool l2r = partner_idx > thr_id;
      const auto diff = loc_data[thr_id] - loc_data[partner_idx];
      const bool to_swap =
          l2r && ((is_left && diff < 0) || (!is_left && diff > 0));
      if (to_swap) {
        swap(loc_data[thr_id], loc_data[partner_idx]);
      }
      __syncthreads();
    }
  }

  data[grid_id] = loc_data[thr_id];
}

int main() {
  int device_count = -1;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return 1;
  }
  if (device_count < 1) {
    return 0;
  }

  cudaDeviceProp device_prop{};
  err = cudaGetDeviceProperties(&device_prop, 0);
  if (err != cudaSuccess) {
    return 1;
  }

  std::cout << device_prop.name << std::endl;
  std::cout << "Device capability: " << device_prop.major << "."
            << device_prop.minor << std::endl;

  return 0;
}