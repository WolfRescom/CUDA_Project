#include <stdio.h>
#include <stdlib.h>
// #include <cuda.h>
#include <chrono>
using namespace std::chrono;

#define numElements 1000000

void radix_sort(int input[], int n) {
  int temp[n];
  for (int currBit = 0; currBit < 32; currBit++) {

    int numZeros = 0;
    for (int i = 0; i < n; i++) {
      if (((input[i] >> currBit) & 1) == 0) {
        numZeros++;
      }
    }

    int zerosIndex = 0;
    int onesIndex = numZeros;

    for (int i = 0; i < n; i++) {
      if (((input[i] >> currBit) & 1) == 0) {
        temp[zerosIndex] = input[i];
        zerosIndex++;
      }
      else {
        temp[onesIndex] = input[i];
        onesIndex++;
      }
    }

    // Need to use some temp array and copy back into input array each time I think, may be unoptimal
    for (int i = 0; i < n; i++) {
      input[i] = temp[i];
    }

  }

}


int main() {
  int input[numElements];

  for (int i = 0; i < numElements; i++) {
    input[i] = rand() % 100;
  }

  // printf("Input array: ");
  // for (int i = 0; i < numElements; i++) {
  //   printf("%d ", input[i]);
  // }
  // printf("\n");

  auto start = high_resolution_clock::now();
  radix_sort(input, numElements);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("CPU radix sort took %ld microseconds\n", duration.count());

  // printf("Sorted array: ");
  // for (int i = 0; i < numElements; i++) {
  //   printf("%d ", input[i]);
  // }
  // printf("\n");

  return 0;
}
