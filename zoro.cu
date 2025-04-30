// ============================
// nbody_cuda.cu (stable spiral galaxy)
// ============================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
}

const int N = 100;
const float G = 6.67430e-11f;
const float dt = 500.0f; // further reduced time step
const int STEPS = 500;

struct Body {
    float x, y;
    float vx, vy;
    float mass;
};

void initialize_spiral_galaxy(std::vector<Body>& bodies, int num_arms = 2, float arm_spread = 0.2f, float galaxy_radius = 5e8f) {
    for (int i = 0; i < bodies.size(); ++i) {
        float t = static_cast<float>(i) / bodies.size();
        float radius = t * galaxy_radius;
        float theta = t * num_arms * 2.0f * M_PI + arm_spread * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        float x = radius * cos(theta);
        float y = radius * sin(theta);

        float enclosed_mass = 1e22f * (i + 1); // approximate enclosed mass
        float vel = sqrt(G * enclosed_mass / (radius + 1e6f));

        float vx = -vel * sin(theta);
        float vy =  vel * cos(theta);

        vx *= 0.9f;
        vy *= 0.9f;

        bodies[i].x = x;
        bodies[i].y = y;
        bodies[i].vx = vx;
        bodies[i].vy = vy;
        bodies[i].mass = 1e22f;
    }
}

__global__ void compute_forces(Body* bodies, float* fx, float* fy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = bodies[i].x, yi = bodies[i].y, mi = bodies[i].mass;
    float fxi = 0.0f, fyi = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        float dx = bodies[j].x - xi;
        float dy = bodies[j].y - yi;
        float distSqr = dx * dx + dy * dy + 1e12f; // increased softening further
        float distSixth = distSqr * sqrtf(distSqr);
        float F = G * mi * bodies[j].mass / distSixth;
        fxi += F * dx;
        fyi += F * dy;
    }
    fx[i] = fxi;
    fy[i] = fyi;
}

__global__ void update_bodies(Body* bodies, float* fx, float* fy, int n, float dt, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ax = fx[i] / bodies[i].mass;
    float ay = fy[i] / bodies[i].mass;
    bodies[i].vx += ax * dt;
    bodies[i].vy += ay * dt;
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;

    if (i == 0 && step % 50 == 0) {
        printf("[step %d] x=%.2f y=%.2f vx=%.5f vy=%.5f ax=%.5e ay=%.5e\n", step, bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy, ax, ay);
    }
}

int main() {
    std::vector<Body> h_bodies(N);
    initialize_spiral_galaxy(h_bodies);

    Body* d_bodies;
    float *d_fx, *d_fy;
    CUDA_CHECK(cudaMalloc(&d_bodies, sizeof(Body) * N));
    CUDA_CHECK(cudaMalloc(&d_fx, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_fy, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies.data(), sizeof(Body) * N, cudaMemcpyHostToDevice));

    std::ofstream out("gpu_output.json");
    out << "[\n";

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int step = 0; step < STEPS; ++step) {
        compute_forces<<<numBlocks, blockSize>>>(d_bodies, d_fx, d_fy, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        update_bodies<<<numBlocks, blockSize>>>(d_bodies, d_fx, d_fy, N, dt, step);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_bodies.data(), d_bodies, sizeof(Body) * N, cudaMemcpyDeviceToHost));

        out << "  [";
        for (int i = 0; i < N; ++i) {
            out << "[" << h_bodies[i].x << "," << h_bodies[i].y << "]";
            if (i < N - 1) out << ",";
        }
        out << "]";
        if (step < STEPS - 1) out << ",";
        out << "\n";
    }
    out << "]\n";

    cudaFree(d_bodies);
    cudaFree(d_fx);
    cudaFree(d_fy);
    return 0;
}

// Compile: nvcc -O2 -o nbody_gpu nbody_cuda.cu
