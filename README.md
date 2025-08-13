# cuda-sandbox

This project implements a [bitonic sorting algorithm](https://en.wikipedia.org/wiki/Bitonic_sorter) using CUDA.
Bitonic sort is a sorting algorithm particularly well-suited for parallel implementation.
The implementation presented here features only the sorting within a single block.

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

