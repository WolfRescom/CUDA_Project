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



__global__ void convolution_1d(int *input, int *output, int *mask, int inputSize, int maskSize) {

  int tid = (blockIdx.x * blockDim.x)+ threadIdx.x;
  if (tid < inputSize) {
    // Calculuate radius of mask so we know how far to look either side
    int radius = maskSize / 2;
    int startPos = tid - radius;

    int runningProduct = 0;
    // iterate across this thread's specific convolution for all elements in the mask in range
    for (int i = 0; i < maskSize; i++) {
      int currIndex = startPos + i;
      if (currIndex >= 0 && currIndex < inputSize) {
        runningProduct += input[currIndex] * mask[i];
      }
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
  int host_input[INPUT_SIZE];
  int host_output[INPUT_SIZE];
  int host_mask[MASK_LENGTH];
  int correct_output[INPUT_SIZE];

  for (int i = 0; i < INPUT_SIZE; i++) {
    host_input[i] = rand() % 100;
  }
  for (int i = 0; i < MASK_LENGTH; i++) {
    host_mask[i] = rand() % 100;
  }

  // Just checking for correctness using the CPU convolution implementation
  convolution_1d_cpu(host_input, correct_output, host_mask, INPUT_SIZE, MASK_LENGTH);

  int *device_input;
  int *device_output;
  int *device_mask;
  cudaMalloc(&device_input, INPUT_SIZE * sizeof(int));
  cudaMalloc(&device_output, INPUT_SIZE * sizeof(int));
  cudaMalloc(&device_mask, MASK_LENGTH * sizeof(int));
  cudaMemcpy(device_input, host_input, INPUT_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_mask, host_mask, MASK_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();
  convolution_1d<<<((INPUT_SIZE + THREADS - 1) / THREADS), THREADS>>>(device_input, device_output, device_mask, INPUT_SIZE, MASK_LENGTH);
  cudaSafeCall(cudaDeviceSynchronize());
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("GPU 1d convolution took %ld microseconds\n", duration.count());

  cudaMemcpy(host_output, device_output, INPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Comparison for correctness
  for (int i = 0; i < INPUT_SIZE; i++) {
    if (correct_output[i] != host_output[i]) {
      printf("Incorrect output at index %d\n", i);
      printf("Correct output: %d\n", correct_output[i]);
    }
  }

  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_mask);
}