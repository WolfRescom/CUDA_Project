#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
using namespace std::chrono;

#define numElements 32768


__global__ void generatePredicate(int* input, int* predicate, int currBit, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Slightly confusing, predicate has 1 if bit is 0, 0 if bit is 1
    predicate[i] = (((input[i] >> currBit) & 1)) == 0;
  }
}

__global__ void upsweep_kernel(int *array, int step) {
  //first compute the current index of the thread being ran to see what element it is working on
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = step * 2;

  //check if the current thread is in scope
  if (i < (numElements / stride)) {
    int currIndex = (i * stride) + (stride - 1);
    //add together LAST 2 NODES using previous stride (step)
    array[currIndex] += array[currIndex - step];
  }
}

__global__ void downsweep_kernel(int *array, int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = step * 2;

  if (i < (numElements / stride)) {
    int currIndex = (i * stride) + (stride - 1);
    int prevIndex = currIndex - step;
    int temp = array[prevIndex];
    array[prevIndex] = array[currIndex];
    array[currIndex] += temp;
  }
}

__global__ void clear_root(int *array, int n) {
  array[n - 1] = 0;
}


__global__ void placeElements(int* input, int* output, int* predicate, int* prefix_sum, int currBit, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Calculate the index we need to insert at
    int numZeros = predicate[n - 1] + prefix_sum[n - 1];
    int index;
    if (predicate[i] == 1) {
      index = prefix_sum[i];
    }
    else {
      index = (i - prefix_sum[i]) + numZeros;
    }
    output[index] = input[i];
  }

}


void radix_sort(int* input, int* output, int n) {
  int* d_input;
  int* d_output;
  cudaMalloc(&d_input, n * sizeof(int));
  cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_output, n * sizeof(int));

  int* d_predicate;
  int* d_prefix;
  cudaMalloc(&d_predicate, n * sizeof(int));
  cudaMalloc(&d_prefix, n * sizeof(int));

  for (int currBit = 0; currBit < 32; currBit++) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 1. Generate predicate array
    generatePredicate<<<blocks, threads>>>(d_input, d_predicate, currBit, n);
    cudaDeviceSynchronize();

    // 2. Calculate prefix sum using predicate
    cudaMemcpy(d_prefix, d_predicate, n * sizeof(int), cudaMemcpyDeviceToDevice);
    for (int i = 1; i < n; i *= 2) {
      upsweep_kernel<<<1, (n / (i * 2))>>>(d_prefix, i);
      cudaDeviceSynchronize();
    }
    clear_root<<<1, 1>>>(d_prefix, n);
    for (int i = (n / 2); i > 0; i /= 2) {
      downsweep_kernel<<<1, (n / (i * 2))>>>(d_prefix, i);
      cudaDeviceSynchronize();
    }

    // 3. Count total number of zeros present
    // int lastPredicate;
    // int lastPrefix;
    // cudaMemcpy(&lastPrefix, d_prefix + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&lastPredicate, d_predicate + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    // int numZeros = lastPredicate + lastPrefix;

    // 4. Compute indices and reorder numElements
    placeElements<<<blocks, threads>>>(d_input, d_output, d_predicate, d_prefix, currBit, n);
    cudaDeviceSynchronize();

    // Swap input and output pointers for next iteration
    std::swap(d_input, d_output);
  }

  cudaMemcpy(output, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}





int main() {
  int input[numElements];
  for (int i = 0; i < numElements; i++) {
    input[i] = rand() % 10;
  }
  int output[numElements];


  // printf("Input array: ");
  //   for (int i = 0; i < numElements; i++) {
  //       printf("%d ", input[i]);
  //   }
  // printf("\n");

  auto start = high_resolution_clock::now();

  radix_sort(input, output, numElements);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("GPU radix sort took %ld microseconds\n", duration.count());


  // printf("Sorted output: ");
  //   for (int i = 0; i < numElements; i++) {
  //       printf("%d ", output[i]);
  //   }
  // printf("\n");

  return 0;
}