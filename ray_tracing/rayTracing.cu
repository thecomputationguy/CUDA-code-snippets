#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/count.h> 
#include <thrust/functional.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric>
#include <time.h> // For srand and time

// --- Global Constants and Error Handling ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define DIM 4096
#define SPHERES 20
#define INF 2e10f

// --- Sphere Structure and Device Method ---
struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float *n) const {
        float dx = ox - x;
        float dy = oy - y;

        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

// --- Constant Memory Declaration (Low-level CUDA) ---
// This array is globally accessible to all device code (functors/kernels).
__constant__ Sphere d_spheres[SPHERES];


// --- RayTracer Functor (Execution Logic) ---
// This functor accesses the scene data directly via the globally defined __constant__ array.

struct raytracer_functor {
    // Note: It only captures DIM and SPHERE count, NOT the sphere data pointer itself.
    const int num_spheres;
    const int dim;

    __host__ __device__ raytracer_functor(int count, int d)
        : num_spheres(count), dim(d) {}

    __device__ unsigned char operator()(int offset) const {
        
        // 1. Map linear offset to coordinates
        int x = offset % dim;
        int y = offset / dim;

        // 2. Map coordinates to world coordinates (ox, oy)
        float ox = (float)(x - dim / 2);
        float oy = (float)(y - dim / 2);

        float r = 0.0f; 
        float maxz = -INF;

        // 3. Ray Tracing Loop over Spheres 
        // Accesses the global __constant__ array defined earlier.
        for (int i = 0; i < num_spheres; i++) {
            Sphere s = d_spheres[i]; // DIRECT ACCESS to constant memory
            float n; 
            float t = s.hit(ox, oy, &n); 

            if (t > maxz) {
                maxz = t;
                float fscale = n; 
                // Only saving R channel for grayscale output
                r = s.r * fscale; 
            }
        }
        
        // 4. Color Mapping (Store Grayscale result)
        return (unsigned char)(r * 255.0f);
    }
};

// --- PPM File Saving Function (Host Code) ---
void save_ppm_file(const std::vector<unsigned char>& image_data_gs, 
                   int width, int height, const char* filename) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the PPM Header (P6 format: binary RGB, 3 bytes per pixel)
    file << "P6\n";
    file << width << " " << height << "\n";
    file << 255 << "\n";

    // Write Raw Pixel Data (RGB: Grayscale = R=G=B)
    for (int i = 0; i < width * height; ++i) {
        unsigned char gs_value = image_data_gs[i];
        file.put(gs_value); // R
        file.put(gs_value); // G
        file.put(gs_value); // B
    }

    file.close();
}


// --- Main Application Logic with CUDA Timing ---
int main() {
    // 1. Setup CUDA Context and Events
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsedTime;
    
    // 2. Host Scene Setup and Initialization
    const int PIXEL_COUNT = DIM * DIM;
    std::vector<Sphere> h_spheres(SPHERES);
    std::vector<unsigned char> h_grayscale_output(PIXEL_COUNT);
    
    srand(static_cast<unsigned int>(time(0)));
    auto rnd = [](float x) { return x * rand() / RAND_MAX; };

    for (int i = 0; i < SPHERES; i++) {
        h_spheres[i].r = rnd(1.0f);
        h_spheres[i].g = rnd(1.0f);
        h_spheres[i].b = rnd(1.0f);
        h_spheres[i].x = rnd(1000.0f) - 500;
        h_spheres[i].y = rnd(1000.0f) - 500;
        h_spheres[i].z = rnd(1000.0f) - 500;
        h_spheres[i].radius = rnd(100.0f) + 20;
    }

    // 3. Device Memory Setup (Input Indices and Output Buffer)
    // Input indices: 0, 1, 2, ... N-1
    thrust::device_vector<int> d_indices(PIXEL_COUNT);
    std::vector<int> h_indices_temp(PIXEL_COUNT);
    std::iota(h_indices_temp.begin(), h_indices_temp.end(), 0);
    d_indices = h_indices_temp; 
    
    unsigned char *d_grayscale_output;
    CUDA_CHECK(cudaMalloc((void**)&d_grayscale_output, PIXEL_COUNT * sizeof(unsigned char)));
    
    // 4. Transfer Scene Data to __constant__ Memory (LOW-LEVEL CUDA)
    // This is the manual step to populate the specialized memory space.
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres, h_spheres.data(), SPHERES * sizeof(Sphere)));

    // 5. Start Timer
    CUDA_CHECK(cudaEventRecord(start, 0));

    // 6. Kernel Execution via Thrust::transform
    raytracer_functor tracer(SPHERES, DIM);
    
    thrust::transform(d_indices.begin(), d_indices.end(), 
                      thrust::device_ptr<unsigned char>(d_grayscale_output), 
                      tracer);

    // 7. Stop Timer and Synchronize
    CUDA_CHECK(cudaEventRecord(stop, 0)); 
    CUDA_CHECK(cudaEventSynchronize(stop)); 
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop)); // Calculate elapsed time

    // 8. Transfer Results Back to Host
    CUDA_CHECK(cudaMemcpy(h_grayscale_output.data(), d_grayscale_output, 
                          PIXEL_COUNT * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // 9. Final Output and Cleanup
    
    const char* filename = "raytracer_output.ppm";
    save_ppm_file(h_grayscale_output, DIM, DIM, filename); 

    std::cout << "Ray tracing complete." << std::endl;
    std::cout << "Time to compute on GPU: " << elapsedTime << " ms" << std::endl;
    std::cout << "Output saved successfully to: " << filename << std::endl;
    
    CUDA_CHECK(cudaFree(d_grayscale_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}