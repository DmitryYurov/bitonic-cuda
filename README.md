# Bitonic Sort

This project implements a [bitonic sorting algorithm](https://en.wikipedia.org/wiki/Bitonic_sorter) using CUDA.
Bitonic sort is a sorting algorithm particularly well-suited for parallel implementation.
The implementation presented here features only the sorting within a single block.

## What does this project do?

In the current implementation, the algorithm sorts a given array of integers in ascending order.
`main` function represents an example of how to use the algorithm, and sorts a vector of 32 integers.
Note that the length of the sorted array must be a power of 2.
The random-generated array is sorted in ascending order and printed to the console.
The maximum size of the array is for now limited by the size of the shared memory per block.

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
