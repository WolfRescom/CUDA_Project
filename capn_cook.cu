#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000  // Size of array
#define THREADS_PER_BLOCK 256

__global__ void countBits(unsigned int *input, int *bitCounts, int bit) {
    __shared__ int localCount[2];  // For bit=0 and bit=1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        localCount[0] = 0;
        localCount[1] = 0;
    }
    __syncthreads();

    if (idx < N) {
        int b = (input[idx] >> bit) & 1;
        atomicAdd(&localCount[b], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&bitCounts[0], localCount[0]);
        atomicAdd(&bitCounts[1], localCount[1]);
    }
}

__global__ void scatter(unsigned int *input, unsigned int *output, int *prefixSums, int bit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int b = (input[idx] >> bit) & 1;
    int pos = atomicAdd(&prefixSums[b], 1);
    output[pos] = input[idx];
}

void radixSortGPU(unsigned int *d_input, unsigned int *d_output) {
    for (int bit = 0; bit < 32; ++bit) {
        int *d_counts, h_counts[2] = {0};
        cudaMalloc(&d_counts, 2 * sizeof(int));
        cudaMemcpy(d_counts, h_counts, 2 * sizeof(int), cudaMemcpyHostToDevice);

        countBits<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input, d_counts, bit);
        cudaMemcpy(h_counts, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        int h_prefix[2] = {0, h_counts[0]};
        int *d_prefix;
        cudaMalloc(&d_prefix, 2 * sizeof(int));
        cudaMemcpy(d_prefix, h_prefix, 2 * sizeof(int), cudaMemcpyHostToDevice);

        scatter<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input, d_output, d_prefix, bit);

        // Swap input and output for next pass
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;

        cudaFree(d_counts);
        cudaFree(d_prefix);
    }
}

void radixSortCPU(unsigned int *arr) {
    unsigned int *output = (unsigned int*)malloc(N * sizeof(unsigned int));
    for (int bit = 0; bit < 32; ++bit) {
        int count[2] = {0};
        for (int i = 0; i < N; ++i) {
            int b = (arr[i] >> bit) & 1;
            count[b]++;
        }
        int prefix[2] = {0, count[0]};
        for (int i = 0; i < N; ++i) {
            int b = (arr[i] >> bit) & 1;
            output[prefix[b]++] = arr[i];
        }
        for (int i = 0; i < N; ++i) arr[i] = output[i];
    }
    free(output);
}

int main() {
    unsigned int *h_input = (unsigned int*)malloc(N * sizeof(unsigned int));
    for (int i = 0; i < N; ++i)
        h_input[i] = rand();

    // CPU timing
    clock_t start_cpu = clock();
    unsigned int *h_cpu = (unsigned int*)malloc(N * sizeof(unsigned int));
    memcpy(h_cpu, h_input, N * sizeof(unsigned int));
    radixSortCPU(h_cpu);
    clock_t end_cpu = clock();
    printf("CPU Time: %.3f ms\n", 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC);

    // GPU setup
    unsigned int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(unsigned int));
    cudaMalloc(&d_output, N * sizeof(unsigned int));
    cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    radixSortGPU(d_input, d_output);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, end_gpu);
    printf("GPU Time: %.3f ms\n", ms);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_cpu);
    return 0;
}
