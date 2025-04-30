#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <chrono>

#define numElements 1000

__global__ void generatePredicate(const int* __restrict__ input,
                                  int* __restrict__ predicate,
                                  int currBit, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // predicate=1 for a 0-bit, 0 for a 1-bit
    predicate[i] = (((input[i] >> currBit) & 1) == 0);
  }
}

__global__ void placeElements(const int* __restrict__ input,
                              int* __restrict__ output,
                              const int* __restrict__ predicate,
                              const int* __restrict__ prefix_sum,
                              int numZeros, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // if bit==0, go to prefix_sum[i]
    // else go to after all zeros: (i - prefix_sum[i]) + numZeros
    int idx = predicate[i]
            ? prefix_sum[i]
            : ( (i - prefix_sum[i]) + numZeros );
    output[idx] = input[i];
  }
}

void radix_sort(int* h_input, int* h_output, int n) {
  int *d_input, *d_output, *d_predicate, *d_prefix;
  cudaMalloc(&d_input,     n * sizeof(int));
  cudaMalloc(&d_output,    n * sizeof(int));
  cudaMalloc(&d_predicate, n * sizeof(int));
  cudaMalloc(&d_prefix,    n * sizeof(int));

  cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

  // timing with CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;

  for (int bit = 0; bit < 32; ++bit) {
    // build predicate array
    generatePredicate<<<blocks, threads>>>(d_input, d_predicate, bit, n);

    // exclusive scan predicate → prefix
    //    (thrust does its own internal sync in the stream)
    thrust::exclusive_scan(thrust::device,
                           d_predicate, d_predicate + n,
                           d_prefix);

    // grab total zeros = last predicate + last prefix
    int lastP, lastPS;
    cudaMemcpy(&lastP,  d_predicate + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastPS, d_prefix    + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int numZeros = lastP + lastPS;

    // scatter into output
    placeElements<<<blocks, threads>>>(d_input, d_output,
                                       d_predicate, d_prefix,
                                       numZeros, n);

    // swap for next bit
    std::swap(d_input, d_output);
  }

  // record and report GPU time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("GPU radix sort took %.3f ms\n", ms);

  // copy result back
  cudaMemcpy(h_output, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_predicate);
  cudaFree(d_prefix);
}

int main() {
  int input[numElements];
  for (int i = 0; i < numElements; i++)
    input[i] = rand() % 100; 

  int output[numElements];

  // printf("Input array: ");
  // for (int i = 0; i < numElements; i++) 
  //   printf("%d ", input[i]);
  // printf("...\n");

  radix_sort(input, output, numElements);

  // printf("Sorted output: ");
  // for (int i = 0; i < 16; i++)
  //   printf("%d ", output[i]);
  // printf("...\n");
  return 0;
}
