#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef struct
{
    float2 pos;
    float2 vel;
    float2 acc;
    float mass;
} Body;

__global__ void update(Body* bodies, int n, float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 zeroAcc = make_float2(0.0f, 0.0f);

    for (int j = 0; j < n; j++) 
    {
        if(j == i) continue;
        float m1 = bodies[i].mass;
        float m2 = bodies[j].mass;

        float dx = bodies[j].pos.x - bodies[i].pos.x;
        float dy = bodies[j].pos.y - bodies[i].pos.y;
        float dist = sqrt(dx * dx + dy * dy);

        //float2 r = make_float2(dx / dist, dy / dist);

        float gravX = dx / (dist * dist * dist);
        float gravY = dy / (dist * dist * dist);

        zeroAcc.x += m2 * gravX;
        zeroAcc.y += m2 * gravY;
    }

    bodies[i].acc = zeroAcc;
    bodies[i].vel.x += zeroAcc.x * dt;
    bodies[i].vel.y += zeroAcc.y * dt;
    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt;
}

//Host Code
int main() 
{
    const int N = 1024; // number of bodies
    const float  dt     = 0.01f;
    const int    steps  = 1000;

    //initializing all the bodies (their positions, velocity, etc.)
    std::vector<Body> hostP(N);
    for(int i = 0; i < N; ++i)
    {
        hostP[i].pos = make_float2(rand()/float(RAND_MAX), rand()/float(RAND_MAX));
        hostP[i].vel = make_float2(0.0f, 0.0f);
        hostP[i].acc = make_float2(0.0f, 0.0f);
        hostP[i].mass = 1.0f;
    }

    
    Body* devP;
    cudaMalloc(&devP, N * sizeof(Body));
    cudaMemcpy(devP, hostP.data(), N * sizeof(Body), cudaMemcpyHostToDevice);


    //threading the needle
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    //runs the GPU kernel "update" for each step 
    for (int t = 0; t < steps; ++t) 
    {
        update<<<blocks, threadsPerBlock>>>(devP, N, dt);
        cudaDeviceSynchronize();
    }

    std::cout << "Body[0] pos = ("
        << hostP[0].pos.x << ", "
        << hostP[0].pos.y << ")\n";


    cudaFree(devP);
    return 0;
}