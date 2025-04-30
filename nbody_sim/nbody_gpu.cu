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

// void initialize_spiral_galaxy(std::vector<Body>& bodies, int num_arms = 2, float arm_spread = 0.5f, float galaxy_radius = 1e9f) {
//     const float center_x = 0.0f;
//     const float center_y = 0.0f;
//     // const float mass_min = 1e22f;
//     const float mass_min = 1e10f;
//     const float mass_variation = 1e5f;

//     for (int i = 0; i < bodies.size(); ++i) {
//         float t = static_cast<float>(i) / bodies.size();
//         float radius = t * galaxy_radius;
//         float angle = t * num_arms * 2.0f * M_PI + arm_spread * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
//         float x = radius * cos(angle);
//         float y = radius * sin(angle);

//         // Circular orbit velocity approximation
//         float dist = sqrt(x * x + y * y);
//         float vel = sqrt(1.0f * 1e24f / (dist + 1e6f)); // Central mass (e.g., black hole at galaxy center)

//         float vx = -vel * sin(angle);
//         float vy = vel * cos(angle);

//         // bodies[i].x = center_x + x;
//         // bodies[i].y = center_y + y;
//         bodies[i].pos = make_float2(x, y);
//         // bodies[i].vx = vx;
//         // bodies[i].vy = vy;
//         bodies[i].vel = make_float2(vx, vy);
//         bodies[i].acc = make_float2(0.0f, 0.0f);
//         bodies[i].mass = mass_min + (rand() % static_cast<int>(mass_variation));
//         // bodies[i].mass = 1.0f;
//     }
// }

__global__ void update(Body* bodies, int n, float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 zeroAcc = make_float2(0.0f, 0.0f);

    for (int j = 0; j < n; j++) 
    {
        if(j == i) continue;
        float m2 = bodies[j].mass;

        float dx = bodies[j].pos.x - bodies[i].pos.x;
        float dy = bodies[j].pos.y - bodies[i].pos.y;
        float dist = sqrt(dx * dx + dy * dy + 0.1f); // avoid div by 0

        float gravX = dx / (dist * dist * dist);
        float gravY = dy / (dist * dist * dist);

        zeroAcc.x += m2 * gravX;
        zeroAcc.y += m2 * gravY;
    }

    bodies[i].acc = zeroAcc;
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
    const int N = 10;
    // const float dt = 0.05f;
    const float dt = 0.005f;
    const int steps = 100;

    std::vector<Body> hostP(N);
    
    // spawn randomly everywhere
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

    // generate spiral galaxy
    spiralGalaxyInit(hostP, N, 50.0f, 50.0f);
    // initialize_spiral_galaxy(hostP);

    // black hole
    // hostP[1023].pos = make_float2(0.05f, 0.05f);
    // hostP[1023].vel = make_float2(0.0f, 0.0f);
    // hostP[1023].acc = make_float2(0.0f, 0.0f);
    // hostP[1023].mass = 1000.0f;


    Body* devP;
    cudaMalloc(&devP, N * sizeof(Body));
    cudaMemcpy(devP, hostP.data(), N * sizeof(Body), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::ofstream file("gpu_output.json");
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

    printf("\nFor %d objects over %d steps...\n", N, steps);
    std::cout << "GPU N-body Simulation time taken: " << duration.count() << " microseconds\n" << std::endl;

    file << "]\n";
    file.close();

    cudaFree(devP);
    return 0;
}
