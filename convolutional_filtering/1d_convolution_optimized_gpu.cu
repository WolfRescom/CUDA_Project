#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
using namespace std::chrono;

// length of mask used for any one step of the convolution, equivalent of 1D kernel
#define MASK_LENGTH 10
#define INPUT_SIZE 1000000
#define THREADS 256

__constant__ int mask[MASK_LENGTH];



#define cudaSafeCall(func) { gpuAssert((func), #func, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *func, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error at %s:%d\n", file, line);
        fprintf(stderr, "  -> %s failed with error: %s\n", func, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}



__global__ void convolution_1d(int *input, int *output, int inputSize, int maskSize) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  extern __shared__ int shared_input[];

  // Calculuate radius of mask so we know how far to look either side
  // int radius = maskSize / 2;

  // 1. First copy input data into shared memory array
  int sharedInputSize = blockDim.x + maskSize;
  int sharedOffset = blockDim.x + threadIdx.x;
  int globalOffset = (blockDim.x * blockIdx.x) + sharedOffset;

  // this copies over data from the input array "radius" bits behind the current
  // threads index due to our padding
  shared_input[threadIdx.x] = input[tid];
  if (sharedOffset < sharedInputSize) {
    shared_input[sharedOffset] = input[globalOffset];
  }

  __syncthreads();

  // 2. Now we can compute the convolution using our fast access shared memory

  if (tid < inputSize) {
    int runningProduct = 0;
    // iterate across this threads specific convolution for all elements in the
    // mask in range
    for (int i = 0; i < maskSize; i++) {
      // Our padding for the shared array is also shifted "radius" bits so
      // no need to index back threadIdx.x by radius
      runningProduct += shared_input[threadIdx.x + i] * mask[i];
    }

    output[tid] = runningProduct;
  }
}



void convolution_1d_cpu(int *input, int *output, int *mask, int inputSize, int maskSize) {
  int radius = maskSize / 2;
  int startPos = 0;

  for (int i = 0; i < inputSize; i++) {
    int runningProduct = 0;
    startPos = i - radius;
    for (int j = 0; j < maskSize; j++) {
      int currIndex = startPos + j;
      if (currIndex >= 0 && currIndex < inputSize) {
        runningProduct += input[currIndex] * mask[j];
      }
    }
    output[i] = runningProduct;
  }
}



int main() {
  // Add padding to the input array for halo values so we can avoid doing bounds checks later
  int PADDED_SIZE = INPUT_SIZE + MASK_LENGTH;
  int radius = MASK_LENGTH / 2;

  int host_input[PADDED_SIZE];
  int host_output[INPUT_SIZE];
  int host_mask[MASK_LENGTH];

  int host_input_copy[INPUT_SIZE];
  int correct_output[INPUT_SIZE];

  // We are padding the edges of the array (the halo) with 0s so 0s dont add to the running product later
  for (int i = 0; i < PADDED_SIZE; i++) {
    if (i < radius || i >= radius + INPUT_SIZE) {
      host_input[i] = 0;
    }
    else {
      int val = rand() % 100;
      host_input[i] = val;
      host_input_copy[i - radius] = val;
    }
  }
  for (int i = 0; i < MASK_LENGTH; i++) {
    host_mask[i] = rand() % 10;
  }

  // Just checking for correctness using the CPU convolution implementation
  convolution_1d_cpu(host_input_copy, correct_output, host_mask, INPUT_SIZE, MASK_LENGTH);

  int *device_input;
  int *device_output;
  cudaMalloc(&device_input, PADDED_SIZE * sizeof(int));
  cudaMalloc(&device_output, INPUT_SIZE * sizeof(int));
  cudaMemcpy(device_input, host_input, PADDED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, host_mask, MASK_LENGTH * sizeof(int));

  int BLOCKS = (INPUT_SIZE + THREADS - 1) / THREADS;
  size_t sharedMemSize = (THREADS + MASK_LENGTH) * sizeof(int);

  auto start = high_resolution_clock::now();
  convolution_1d<<<BLOCKS, THREADS, sharedMemSize>>>(device_input, device_output, INPUT_SIZE, MASK_LENGTH);
  cudaSafeCall(cudaDeviceSynchronize());
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("\nFor input array of size %d and mask array of size %d\n", INPUT_SIZE, MASK_LENGTH);
  printf("GPU 1d optimized convolution took %ld microseconds\n\n", duration.count());

  cudaMemcpy(host_output, device_output, INPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Comparison for correctness
  // for (int i = 0; i < INPUT_SIZE; i++) {
  //   if (correct_output[i] != host_output[i]) {
  //     printf("Incorrect output at index %d\n", i);
  //     printf("Correct output: %d\n", correct_output[i]);
  //     printf("Correct output: %d\n", host_output[i]);
  //   }
  // }

  cudaFree(device_input);
  cudaFree(device_output);
}