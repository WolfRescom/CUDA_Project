#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <iomanip>  // for std::setprecision

#include <chrono>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
}

typedef struct
{
    float2 pos; 
    float2 vel;
    float2 acc;
    float mass;
} Body;

void spiralGalaxyInit(std::vector<Body>& bodies, int N, float centerX, float centerY, float galaxyRadius = 50.0f, int arms = 2, float spread = 0.5f, float mass = 1.0f)
{
    for (int i = 0; i < N; ++i) {
        float angle = ((float)i / N) * arms * 2.0f * M_PI;  // spiral arms
        float radius = galaxyRadius * sqrt((float)rand() / RAND_MAX); // more stars near center

        // Add spread using random noise
        float noise = spread * ((float)rand() / RAND_MAX - 0.5f);

        // Final angle per body with twist
        float theta = angle + noise;

        float x = centerX + radius * cos(theta);
        float y = centerY + radius * sin(theta);

        // Velocity perpendicular to radius vector (tangential)
        float vx = -sin(theta);  // orthogonal unit vector
        float vy = cos(theta);
        float speed = sqrt(1.0f / (radius + 0.1f)); // approx orbital speed using G*M/r, simplified

        bodies[i].pos = make_float2(x, y);
        bodies[i].vel = make_float2(vx * speed, vy * speed);
        bodies[i].acc = make_float2(0.0f, 0.0f);
        bodies[i].mass = mass;
    }
}

__global__ void update(Body* bodies, int n, float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 zeroAcc = make_float2(0.0f, 0.0f);

    for (int j = 0; j < n; j++) 
    {
        if(j == i) continue;
        //float m1 = bodies[i].mass;
        float m2 = bodies[j].mass;

        float dx = bodies[j].pos.x - bodies[i].pos.x;
        float dy = bodies[j].pos.y - bodies[i].pos.y;
        float dist = sqrt(dx * dx + dy * dy + 5e-3f); // avoid div by 0

        float gravX = dx / (dist * dist * dist);
        float gravY = dy / (dist * dist * dist);

        zeroAcc.x += m2 * gravX;
        zeroAcc.y += m2 * gravY;
    }

    bodies[i].acc = zeroAcc;
    // bodies[i].vel.x += zeroAcc.x * dt;
    // bodies[i].vel.y += zeroAcc.y * dt;
    // bodies[i].pos.x += bodies[i].vel.x * dt;
    // bodies[i].pos.y += bodies[i].vel.y * dt; 
}

__global__ void update_bodies(Body* bodies, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 zeroAcc = bodies[i].acc;
    bodies[i].vel.x += zeroAcc.x * dt;
    bodies[i].vel.y += zeroAcc.y * dt;
    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt; 
}

int main() 
{
    const int N = 64;
    // const float dt = 0.05f;
    const float dt = 0.01f;
    const int steps = 1000;

    std::vector<Body> hostP(N);
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {

        }
    }

    // for(int i = 0; i < N; ++i) 
    // {
    //     // hostP[i].pos = make_float2(
    //     //     (rand()/float(RAND_MAX)) / 0.1 + 0.05, 
    //     //     (rand()/float(RAND_MAX)) / 0.1 + 0.05
    //     // );
    //     // hostP[i].vel = make_float2(
    //     //     (rand()/float(RAND_MAX)) / 10, 
    //     //     (rand()/float(RAND_MAX)) / 10
    //     // );;
    //     // hostP[i].vel = make_float2(rand()/float(RAND_MAX), rand()/float(RAND_MAX));
    //     hostP[i].vel = make_float2(0.0f, 0.0f);
    //     hostP[i].acc = make_float2(0.0f, 0.0f);
    //     hostP[i].mass = 1.0f;
    // }

    // spiralGalaxyInit(hostP, N, 50.0f, 50.0f);  // center at (50, 50), adjust as needed

    // hostP[1023].pos = make_float2(0.05f, 0.05f);
    // hostP[1023].vel = make_float2(0.0f, 0.0f);
    // hostP[1023].acc = make_float2(0.0f, 0.0f);
    // hostP[1023].mass = 1000.0f;


    Body* devP;
    cudaMalloc(&devP, N * sizeof(Body));
    cudaMemcpy(devP, hostP.data(), N * sizeof(Body), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::ofstream file("output.json");
    file << std::fixed << std::setprecision(5);  // control float format
    file << "[\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < steps; ++t) 
    {
        update<<<blocks, threadsPerBlock>>>(devP, N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        update_bodies<<<blocks, threadsPerBlock>>>(devP, N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaMemcpy(hostP.data(), devP, N * sizeof(Body), cudaMemcpyDeviceToHost);

        file << "  [";
        for (int i = 0; i < N; ++i) {
            file << "[" << hostP[i].pos.x << "," << hostP[i].pos.y << "]";
            if (i < N - 1) file << ",";
        }
        file << "]";
        if (t < steps - 1) file << ",";
        file << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU N-body Simulation time taken: " << duration.count() << " microseconds" << std::endl;

    file << "]\n";
    file.close();

    cudaFree(devP);
    return 0;
}
