#include "../includes/bitonic.cuh"

namespace {
/**
 * @brief Device-side swap implementation
 */
__device__ void swap(int &lhs, int &rhs) noexcept {
  const int tmp = lhs;
  lhs = rhs;
  rhs = tmp;
}

/**
 * @brief Determines if two values need to be swapped during the sort based on tonic size, index of the left-hand-side element
 *        and the values in the array
 *
 * @param data_index Index in the data array to compare with the tonic size. This is the index of the lhs entry in the
 *                   original data sequence.
 * @param tonic_size the size of the interval being sorted
 * @param lhs Left-hand side value to compare. This value corresponds to a smaller index in the data array, the index itself
 *            is passed with data_index input argument.
 * @param rhs Right-hand side value to compare. This value corresponds to a larger index in the data array.
 * @return True if the values need a swap, false otherwise
 */
__device__ bool needs_swap(size_t data_index, size_t tonic_size, int lhs, int rhs) noexcept {
  // first find out if we are on an ascending (left) or descending (right) slope of the bitonic sequence
  const bool ascending = (data_index & tonic_size) == 0;
  return (ascending && (lhs > rhs) || (!ascending && lhs < rhs));
}

/**
 * @brief Part of the bitonic sort performed on shared data of the thread block.
 *
 * @param data Device-allocated array to be sorted
 * @note Assumes the size of the input array is a power of 2 and equal to the block size.
 * @note Each thread block processes blockDim.x consecutive elements from the input array
 * @note Uses shared memory of size blockDim.x * sizeof(int) for local sorting
 */
__global__ void bitonic_shared(int *data) {
  extern __shared__ int loc_data[];
  const unsigned thr_id = threadIdx.x;
  const unsigned data_id = blockIdx.x * blockDim.x + thr_id;

  loc_data[thr_id] = data[data_id];
  __syncthreads();

  // the outer loop defines the size of the interval (tonic) for sorting the values in ascending or descending order
  for (unsigned k = 2U; k <= blockDim.x; k <<= 1U) {
    // in the inner loop we choose a comparison interval and swap elements with their partners if necessary
    for (unsigned j = k >> 1U; j > 0U; j >>= 1U) {
      // partner_id must always be larger than thr_id, otherwise we could encounter a race of swaps.
      const unsigned partner_id = thr_id | j;
      if (needs_swap(data_id, k, loc_data[thr_id], loc_data[partner_id])) {
        swap(loc_data[thr_id], loc_data[partner_id]);
      }
      __syncthreads();
    }
  }

  data[data_id] = loc_data[thr_id];
}

/**
 * @brief Single iteration of the bitonic sort algorithm using global memory
 *
 * @param data Device-allocated array being sorted
 * @param tonic_size Size of the current bitonic sequence being sorted
 * @param stride Distance between elements being compared and potentially swapped
 * @note This kernel is launched with grid_size blocks, each containing block_size threads
 * @note Each thread processes one element of the array by comparing and potentially swapping
 *       it with its partner element determined by the stride
 */
__global__ void iteration(int *data, unsigned tonic_size, unsigned stride) {
  const unsigned data_id = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned partner_idx = data_id | stride;
  if (needs_swap(data_id, tonic_size, data[data_id], data[partner_idx])) {
    swap(data[data_id], data[partner_idx]);
  }
}

/**
 * @brief Part of the bitonic sort using global memory for larger arrays or final stages of sorting.
 *        This function handles the steps after the initial shared memory sorting to merge sorted sequences further.
 *
 * @param data Device-allocated array to be sorted
 * @param data_size Size of the array to sort (must be power of 2)
 * @param grid_size Number of thread blocks to launch
 * @param block_size Number of threads per block
 * @note It is assumed that the input data is already sorted up to the tonics of block_size
 */
__host__ void bitonic_global(int *data, unsigned data_size, unsigned grid_size, unsigned block_size) {
  unsigned size = block_size << 1U;
  while (size <= data_size) {
    for (unsigned stride = size >> 1U; stride > 0U; stride >>= 1U) {
      iteration<<<grid_size, block_size>>>(data, size, stride);
    }
    size <<= 1U;
  }
}

/**
 * @brief Determines the size of the CUDA execution block as the minimum of the three values:
 *        - max number of threads per block
 *        - number of elements in the array to sort
 *        - number of elements which can be accommodated in the shared memory of the block
 *
 * @param data_size The number of elements in the array to sort. Must be a power of 2
 * @param max_threads_per_block Maximum number of threads per CUDA execution block
 * @param shared_mem_per_block Shared memory in bytes per CUDA execution block
 * @return The size of the CUDA thread block
 * @note Implicitly assumed type of the array element is int.
 * @note The size of the block is selected in such a way, that bitonic_shared kernel can be executed conveniently.
 */
unsigned get_block_size(unsigned data_size, unsigned max_threads_per_block, unsigned shared_mem_per_block) {
  using namespace bitonic::utilities;
  // we assume that data_size is already a power of two
  const size_t limit = std::min(floor_2(max_threads_per_block), floor_2(shared_mem_per_block / sizeof(int)));
  return static_cast<unsigned>(std::min(limit, static_cast<size_t>(data_size)));
}
} // anonymous namespace

namespace bitonic {
__host__ void sort(int *d_data, unsigned data_size, const cudaDeviceProp &properties) {
  // finding the block size
  const unsigned block_size = get_block_size(data_size, properties.maxThreadsPerBlock, properties.sharedMemPerBlock);
  const unsigned grid_size = data_size / block_size;

  bitonic_shared<<<grid_size, block_size, block_size * sizeof(int)>>>(d_data);
  if (grid_size > 1U) {
    bitonic_global(d_data, data_size, grid_size, block_size);
  }
}

namespace utilities {
size_t ceil_2(unsigned val) {
  if (val <= 1U) {
    return 1U;
  }

  size_t res = 2U;
  while (res < val) {
    res <<= 1U;
  }

  return res;
}

size_t floor_2(unsigned val) {
  const size_t res = ceil_2(val);

  if (res < 2U) {
    return res;
  }
  return res == val ? res : res >> 1U;
}
} // namespace utilities
} // namespace bitonic
