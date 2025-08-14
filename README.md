# Bitonic Sort

This project implements a [bitonic sorting algorithm](https://en.wikipedia.org/wiki/Bitonic_sorter) using CUDA.
Bitonic sort is a sorting algorithm particularly well-suited for parallel implementation.

## What does this project do?

The algorithm sorts a given array of integers in ascending order.

The implementation contains two main functions:

- `bitonic_shared`: Performs sorting using shared memory (smaller arrays or initial stages of sorting)
- `bitonic_global`: Performs sorting using global memory (larger arrays or final stages of sorting)

The `run_sort` function orchestrates the sorting process by:

- Calling the bitonic sort implementation
- Measuring execution time
- Comparing results with `std::sort`

The `main` function tests the implementation for a number of array sizes.

## Requirements

- CMake 3.26 or higher
- CUDA-capable GPU
- CUDA Toolkit

## Building

```bash
mkdir build 
cd ./build
cmake -S .. -B .
cmake --build .
```
