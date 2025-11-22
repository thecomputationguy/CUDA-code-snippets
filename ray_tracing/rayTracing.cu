#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric>

// --- Constants & Global Definitions ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define DIM 512
#define SPHERES 20
#define INF 2e10f

// --- Sphere Structure and Device Method (from Chapter 6) ---

struct Sphere {
    float r, g, b;      // Color components
    float radius;
    float x, y, z;      // Center coordinates

    // __device__ method to calculate intersection point (t) and normal (n)
    __device__ float hit(float ox, float oy, float *n) const {
        float dx = ox - x;
        float dy = oy - y;

        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            // This 'n' is used for shading scale in the original code, 
            // representing depth/radius for shading (0 to 1).
            *n = dz / sqrtf(radius * radius); 
            return dz + z; // Return depth (z + dz)
        }
        return -INF; // No hit
    }
};

// --- RayTracer Functor (The Kernel Logic) ---
// This functor will be executed once per pixel by thrust::transform.

struct raytracer_functor {
    // We pass the entire Sphere array via a raw device pointer (C-style array captured by value)
    Sphere* d_spheres;
    const int num_spheres;
    const int dim;

    __host__ __device__ raytracer_functor(Sphere* spheres, int count, int d)
        : d_spheres(spheres), num_spheres(count), dim(d) {}

    // Operator to apply the ray-tracing calculation logic to a single linear index
    __device__ unsigned char operator()(int offset) const {
        // 1. Map linear offset to pixel coordinates (x, y)
        int x = offset % dim;
        int y = offset / dim;

        // 2. Map pixel coordinates to world coordinates (ox, oy)
        float ox = (float)(x - dim / 2);
        float oy = (float)(y - dim / 2);

        float r = 0.0f, g = 0.0f, b = 0.0f;
        float maxz = -INF; // Tracks the closest object hit

        // 3. Iterate through all spheres (The core ray tracing loop)
        for (int i = 0; i < num_spheres; i++) {
            // Access the sphere data using the device pointer
            Sphere s = d_spheres[i]; 
            float n; // Stores the calculated depth/shading factor
            
            // Check for intersection
            float t = s.hit(ox, oy, &n); 

            // If a hit occurred and it's closer than the current closest object
            if (t > maxz) {
                // Update closest object and color
                maxz = t;
                float fscale = n; 
                r = s.r * fscale;
                g = s.g * fscale;
                b = s.b * fscale;
            }
        }
        
        // 4. Color Mapping to Single Byte (Grayscale)
        // Since we are only saving a PPM, we'll convert the color result to a single grayscale byte (0-255).
        // The original code used 4 bytes for RGBA, this simplifies output to 3 bytes (RGB) in PPM.
        return (unsigned char)(r * 255.0f); // Use R channel as grayscale output
    }
};

// --- PPM File Saving Function (Accepts std::vector) ---
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

    // Write Raw Pixel Data (RGB)
    // Since the image_data is grayscale (1 byte per pixel), we write R, G, B as the same value.
    for (int i = 0; i < width * height; ++i) {
        unsigned char gs_value = image_data_gs[i];
        file.put(gs_value); // R
        file.put(gs_value); // G
        file.put(gs_value); // B
    }

    file.close();
}

// --- Main Application Logic with Thrust ---
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    const int N = DIM * DIM;
    
    // --- 1. Host Scene Setup ---
    std::vector<Sphere> h_spheres(SPHERES);
    
    // Random number generator setup (using host logic from Chapter 6)
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

    // --- 2. Device Setup and Data Transfer (Scene Data) ---
    
    // Use thrust::device_vector to manage the sphere data (which is constant for the kernel run)
    thrust::device_vector<Sphere> d_spheres_vector = h_spheres;
    // Get a raw pointer to the device memory for the functor to capture
    Sphere* d_spheres_ptr = thrust::raw_pointer_cast(d_spheres_vector.data());


    // --- 3. Execution Setup ---
    
    // Input indices: 0, 1, 2, ... N-1
    thrust::device_vector<int> d_indices(N);
    std::vector<int> h_indices_temp(N);
    std::iota(h_indices_temp.begin(), h_indices_temp.end(), 0);
    d_indices = h_indices_temp; 
    
    // Output: 1 byte per pixel (Grayscale R channel)
    thrust::device_vector<unsigned char> d_grayscale_output(N);

    // Create the functor, capturing the device pointer to the sphere data
    raytracer_functor tracer(d_spheres_ptr, SPHERES, DIM);
    
    // 4. Execute Ray Tracing via Thrust::transform
    // The device iterators implicitly determine the execution policy.
    std::cout << "Starting ray tracing on GPU (" << DIM << "x" << DIM << " pixels)..." << std::endl;

    thrust::transform(d_indices.begin(), d_indices.end(), 
                      d_grayscale_output.begin(), tracer);

    // 5. Transfer Results back to Host
    thrust::host_vector<unsigned char> h_grayscale_output = d_grayscale_output;
    
    // Convert to standard vector for file saving
    std::vector<unsigned char> h_output_for_file(h_grayscale_output.begin(), h_grayscale_output.end());

    // 6. Save Final Output
    const char* filename = "raytracer_output.ppm";
    save_ppm_file(h_output_for_file, DIM, DIM, filename); 

    std::cout << "Ray tracing complete." << std::endl;
    std::cout << "Output saved successfully to: " << filename << std::endl;
    
    return 0;
}