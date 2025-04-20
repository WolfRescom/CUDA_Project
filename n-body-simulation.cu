#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef struct{
    float2 pos;
    float2 vel;
    float2 acc;
    float mass;
} Body;


typedef struct Simulation{

    int n;
    Body[] bodies;

    __global__ update() {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float2 p1 = bodies[i].pos;
                float m1 = bodies[i].mass;
                float2 p2 = bodies[j].pos;
                float m2 = bodies[j].mass;

                float dx = bodies[j].pos.x - bodies[i].pos.x;
                float dy = bodies[j].pos.y - bodies[i].pos.y;
                float dist = sqrt(dx * dx + dy * dy);

                float2 r = make_float2(dx / dist, dy / dist);

                float grav = r / (dist * dist * dist);

                bodies[i].acc += m2 * grav;
                bodies[j].acc -= m1 * grav;
            }
        }
    }


}

//Host Code
int main() {
    const int N = 1024; // number of bodies
    std::vector<Body> hostP(N);
    for(int i = 0; i < N; ++i){
        hostP[i].pos = make_float2(rand()/float(RAND_MAX), rand()/float(RAND_MAX));
        hostP[i].vel = make_float2(0.0f, 0.0f);
        hostP[i].acc = make_float2(0.0f, 0.0f);
        hostP[i].mass = 1.0f;
    }


}