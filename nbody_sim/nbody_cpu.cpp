// ========================
// nbody_cpu.cpp (basic C++)
// ========================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <chrono>

const int N = 10000;
const float G = 6.67430e-11f;
// const float dt = 1e3f;
const float dt = 0.01f;
// const int STEPS = 500;
const int STEPS = 100;

struct Body {
    float x, y;
    float vx, vy;
    float mass;
};

void initialize_spiral_galaxy(std::vector<Body>& bodies, int num_arms = 2, float arm_spread = 0.5f, float galaxy_radius = 1e9f) {
    const float center_x = 0.0f;
    const float center_y = 0.0f;
    const float mass_min = 1e22f;
    const float mass_variation = 1e5f;

    for (int i = 0; i < bodies.size(); ++i) {
        float t = static_cast<float>(i) / bodies.size();
        float radius = t * galaxy_radius;
        float angle = t * num_arms * 2.0f * M_PI + arm_spread * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        float x = radius * cos(angle);
        float y = radius * sin(angle);

        // Circular orbit velocity approximation
        float dist = sqrt(x * x + y * y);
        float vel = sqrt(G * 1e24f / (dist + 1e6f)); // Central mass (e.g., black hole at galaxy center)

        float vx = -vel * sin(angle);
        float vy = vel * cos(angle);

        bodies[i].x = center_x + x;
        bodies[i].y = center_y + y;
        bodies[i].vx = vx;
        bodies[i].vy = vy;
        bodies[i].mass = mass_min + (rand() % static_cast<int>(mass_variation));
    }
}


int main() {
    std::vector<Body> bodies(N);
    for (auto& b : bodies) {
        b.x = (rand() % 2000000000 - 1000000000);
        b.y = (rand() % 2000000000 - 1000000000);
        b.vx = 0.0f;
        b.vy = 0.0f;
        b.mass = 1e22f + (rand() % 100000);

        // initialize_spiral_galaxy(bodies);
    } // shh i'll keep it a secret bro

    std::ofstream out("cpu_output.json");
    out << "[\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < STEPS; ++step) {
        std::vector<float> fx(N, 0), fy(N, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float distSqr = dx * dx + dy * dy + 1e6f;
                float distSixth = distSqr * sqrtf(distSqr);
                float F = G * bodies[i].mass * bodies[j].mass / distSixth;
                fx[i] += F * dx;
                fy[i] += F * dy;
            }
        }

        for (int i = 0; i < N; ++i) {
            bodies[i].vx += fx[i] / bodies[i].mass * dt;
            bodies[i].vy += fy[i] / bodies[i].mass * dt;
            bodies[i].x += bodies[i].vx * dt;
            bodies[i].y += bodies[i].vy * dt;
        }

        out << "  [";
        for (int i = 0; i < N; ++i) {
            out << "[" << bodies[i].x << "," << bodies[i].y << "]";
            if (i < N - 1) out << ",";
        }
        out << "]";
        if (step < STEPS - 1) out << ",";
        out << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("\nFor %d objects over %d steps...\n", N, STEPS);
    std::cout << "CPU N-body Simulation time taken: " << duration.count() << " microseconds\n" << std::endl;

    out << "]\n";
    return 0;
}