#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

namespace bitonic {
/**
 * @brief A wrapper function for performing bitonic sort on a CUDA-capable GPU.
 *
 *        The routine internally performs the computation of block and grid sizes, and runs the sort
 *        on a currently selected device. It's the responsibility of the user to allocate
 *        and initialize the device-side array of input values.
 *
 * @param d_data Device-allocated data to sort.
 * @param data_size The size of the array. Note that the size of the input array must be a power of 2.
 * @param properties Device properties used to compute block and grid sizes
 */
__host__ void sort(int *d_data, unsigned data_size, const cudaDeviceProp &properties);
} // namespace bitonic

#endif // BITONIC_SORT_CUH
