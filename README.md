# Bitonic Sort

This project implements the [bitonic sorting algorithm](https://en.wikipedia.org/wiki/Bitonic_sorter) using CUDA.
Bitonic sort is a sorting algorithm particularly well-suited for parallel implementation.

## What does this project do?

The `main` function initializes a CUDA device and executes the bitonic sort algorithm on it.
Additionally, it verifies the correctness of the results by comparing them with `std::sort`,
and prints the execution times for both bitonic sort and `std::sort`.

The algorithm sorts a given array of integers in ascending order in two main stages:

1. Initial sorting using shared memory within each thread block
2. Merging sorted sequences using global memory operations (for the arrays which do not fit in shared memory)

Input array size must always be a power of 2 (a limitation of the algorithm).

## Project structure

1. `include` - header files. Currently, only `bitonic.cuh` is provided, which contains the declarations of the bitonic sort
               wrapper as well as some helper functions.
2. `src` - source files. `bitonic.cu` contains the implementation of the bitonic sort, while `main.cu` contains
           all the utilities necessary for running the sample program.

All the source files and headers are thoroughly documented. Please refer to the built-in documentation in case you want
to investigate the implementation details.

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

## Running

If run without arguments, the program will compare the results of bitonic sort with `std::sort` for a number of
array sizes (from `256` to `256 * 2^18`):

```bash
cd ./build
./bitonic_sort
```

An example output for NVIDIA GeForce GTX 1660 Ti and Intel Core i7-9750H CPU is shown below:

```
NVIDIA GeForce GTX 1660 Ti
Device capability: 7.5
Data size  | GPU time   | CPU time
       1 kB     0.25 ms     0.01 ms
       2 kB     0.17 ms     0.02 ms
       4 kB     0.16 ms     0.04 ms
       8 kB     0.20 ms     0.08 ms
      16 kB     0.22 ms     0.17 ms
      32 kB     0.26 ms     0.36 ms
      64 kB     0.29 ms     0.78 ms
     128 kB     0.43 ms     1.69 ms
     256 kB     0.87 ms     3.64 ms
     512 kB     0.82 ms     7.81 ms
    1024 kB     1.43 ms    16.56 ms
    2048 kB     3.50 ms    35.39 ms
    4096 kB     7.00 ms    74.67 ms
    8192 kB    14.81 ms   157.83 ms
   16384 kB    32.00 ms   326.89 ms
   32768 kB    69.85 ms   693.16 ms
   65536 kB   152.75 ms  1422.30 ms
  131072 kB   282.02 ms  3052.01 ms
  262144 kB   748.82 ms  6356.76 ms
```

A user can also specify the size of the array to sort:

```bash
cd ./build
./bitonic_sort 1024
```

In this case, the program will execute the sorting and comparison for a single array with the specified number of
elements:

```
NVIDIA GeForce GTX 1660 Ti
Device capability: 7.5
Data size  | GPU time   | CPU time
       4 kB     0.23 ms     0.04 ms
```
