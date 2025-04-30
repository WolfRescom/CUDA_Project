#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define MASK_DIM 3
#define MASK_OFFSET (MASK_DIM / 2)

// Constant memory for mask
__constant__ int mask[MASK_DIM * MASK_DIM];

// CUDA Kernel: 2D Convolution
__global__ void convolution_2d(unsigned char *input, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int temp = 0;

        for (int i = 0; i < MASK_DIM; i++) {
            for (int j = 0; j < MASK_DIM; j++) {
                int r = row + i - MASK_OFFSET;
                int c = col + j - MASK_OFFSET;

                // Boundary check
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    temp += (int)input[r * width + c] * mask[i * MASK_DIM + j];
                }
            }
        }

        // Normalize (for blur filter)
        temp = temp / 9;

        // Clamp to [0, 255]
        temp = temp > 255 ? 255 : (temp < 0 ? 0 : temp);

        output[row * width + col] = (unsigned char) temp;

        // Optional: Print first pixel for debugging
        if (row == 0 && col == 0) {
            printf("First input pixel = %d\n", input[0]);
            printf("First output pixel = %d\n", temp);
        }
    }
}

// Helper function to load .bin file
void load_image(const char *filename, unsigned char *data, size_t size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Failed to open input file!\n");
        exit(1);
    }
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);
}

// Helper function to save .bin file
void save_image(const char *filename, unsigned char *data, size_t size) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to open output file!\n");
        exit(1);
    }
    fwrite(data, sizeof(unsigned char), size, f);
    fclose(f);
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const size_t image_size = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char *h_input = (unsigned char *)malloc(image_size);
    unsigned char *h_output = (unsigned char *)malloc(image_size);

    // Load input image (.bin file)
    load_image("input_image.bin", h_input, width * height);

    // Define 3x3 blur mask
    int h_mask[MASK_DIM * MASK_DIM] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    // Copy input image and mask to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, MASK_DIM * MASK_DIM * sizeof(int));

    // Configure block and grid sizes
    int THREADS = 16;
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim((width + THREADS - 1) / THREADS, (height + THREADS - 1) / THREADS);

    // Launch kernel
    convolution_2d<<<grid_dim, block_dim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Save output image
    save_image("output_image.bin", h_output, width * height);

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("COMPLETED SUCCESSFULLY!\n");

    return 0;
}
