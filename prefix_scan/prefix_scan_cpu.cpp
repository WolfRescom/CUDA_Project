#include <stdio.h>
#include <stdlib.h>
// #include <cuda.h>
#include <chrono>
using namespace std::chrono;

// #define numElements 524288
#define numElements 1000000

void prefix_scan(int input[], int result[], int n) {
  result[0] = 0;
  for (int i = 1; i < n; i++) {
    result[i] = result[i - 1] + input[i - 1];
  }
}

int main() {
  int input[numElements];
  int output[numElements];

  for (int i = 0; i < numElements; i++) {
    input[i] = rand() % 10;
  }

  auto start = high_resolution_clock::now();
  prefix_scan(input, output, numElements);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("\nFor array with %d elements...\n", numElements);
  printf("CPU prefix scan took %ld microseconds\n\n", duration.count());

  return 0;
}