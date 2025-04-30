#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// #include <cuda.h>
#include <chrono>
using namespace std::chrono;

// length of mask used for any one step of the convolution, equivalent of 1D kernel
#define MASK_LENGTH 10
#define INPUT_SIZE 1000000
#define THREADS 256


void convolution_1d_cpu(int *input, int *output, int *mask,
                        int inputSize, int maskSize) {
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
  int input[INPUT_SIZE];
  int output[INPUT_SIZE];
  int mask[MASK_LENGTH];

  for (int i = 0; i < INPUT_SIZE; i++) {
    input[i] = rand() % 100;
  }
  for (int i = 0; i < MASK_LENGTH; i++) {
    mask[i] = rand() % 100;
  }

  // printf("Input array: ");
  // for (int i = 0; i < 20; i++) {
  //   printf("%d ", input[i]);
  // }
  // printf("\n");

  // printf("Mask array: ");
  // for (int i = 0; i < MASK_LENGTH; i++) {
  //   printf("%d ", mask[i]);
  // }
  // printf("\n");

  auto start = high_resolution_clock::now();
  convolution_1d_cpu(input, output, mask, INPUT_SIZE, MASK_LENGTH);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("\nFor input array of size %d and mask array of size %d\n", INPUT_SIZE, MASK_LENGTH);
  printf("CPU 1d convolution took %ld microseconds\n\n", duration.count());

  // printf("Convolution output: ");
  // for (int i = 0; i < 20; i++) {
  //   printf("%d ", output[i]);
  // }
  // printf("\n");

  return 0;
}