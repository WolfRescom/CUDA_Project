#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
using namespace std::chrono;

// #define numElements 524288
#define numElements 1000000

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


__global__ void clear_root(int *array) {
  array[numElements - 1] = 0;
}


int main() {
  int host_array[numElements];

  for (int i = 0; i < numElements; i++) {
    host_array[i] = rand() % 10;
  }

  int *device_array;
  cudaMalloc(&device_array, numElements * sizeof(int));
  cudaMemcpy(device_array, host_array, numElements * sizeof(int), cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();
  for (int i = 1; i < numElements; i *= 2) {
    upsweep_kernel<<<1, (numElements / (i * 2))>>>(device_array, i);
    //make sure we synchronize threads before moving to next step
    cudaDeviceSynchronize();
  }

  // now set root node back to 0
  clear_root<<<1, 1>>>(device_array);

  for (int i = (numElements / 2); i > 0; i /= 2) {
    downsweep_kernel<<<1, (numElements / (i * 2))>>>(device_array, i);
    cudaDeviceSynchronize();
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("\nFor array with %d elements...\n", numElements);
  printf("GPU prefix scan took %ld microseconds\n\n", duration.count());


  cudaMemcpy(host_array, device_array, numElements * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_array);

  return 0;
}