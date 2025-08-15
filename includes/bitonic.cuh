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

namespace utilities {
/**
 * @brief Rounds value up to the nearest non-negative power of 2 that is greater than or equal to the input value
 *
 * @param val Input value to round up
 * @return size_t Value rounded up to the nearest non-negative power of 2
 */
size_t ceil_2(unsigned val);

/**
 * @brief Rounds value down to the nearest non-negative power of 2 such that the result is less or equal to the input value.
 *        Returns 1 (2^0) in case val <= 1
 *
 * @param val Input value to round down
 * @return size_t Value rounded down to the nearest power of 2, never returns zero
 */
size_t floor_2(unsigned val);

} // namespace utilities
} // namespace bitonic

#endif // BITONIC_SORT_CUH
