#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/count.h> 
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric> // Required for std::iota (used to create indices)
#include <algorithm> // Required for std::count (used for verification)

// --- Helper for native CUDA error handling ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define DIM 1000
#define ITERATIONS 200
#define MAG_THRESHOLD 1000

// --- Complex Number Structure and Operations ---
struct cuComplex {
    float r;
    float i;
    __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2() const { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a) const { return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i); }
    __device__ cuComplex operator+(const cuComplex& a) const { return cuComplex(r + a.r, i + a.i); }
};

// --- Julia Set Functor (Device Code) ---
struct julia_functor {
    const float scale;
    const cuComplex c_const;
    const int dim;

    __host__ __device__ julia_functor(int d) 
        : scale(1.5f), c_const(-0.8f, 0.156f), dim(d) {}

    __device__ unsigned char operator()(int offset) const {
        int x = offset % dim;
        int y = offset / dim;

        float jx = scale * (float)(dim / 2 - x) / (dim / 2);
        float jy = scale * (float)(dim / 2 - y) / (dim / 2);
        
        cuComplex a(jx, jy);
        
        for (int i = 0; i < ITERATIONS; i++) {
            a = a * a + c_const;
            if (a.magnitude2() > MAG_THRESHOLD) { return 0; } 
        }
        return 1; 
    }
};

// --- PPM File Saving Function ---
void save_ppm_file(const std::vector<unsigned char>& image_data_rgba, 
                   int width, int height, const char* filename) {
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the PPM Header (P6 format: binary RGB)
    file << "P6\n";
    file << width << " " << height << "\n";
    file << 255 << "\n"; // Max color value

    // Write Raw Pixel Data (RGB)
    const int RGBA_COMPONENTS = 4;
    
    for (int i = 0; i < width * height; ++i) {
        int offset_rgba = i * RGBA_COMPONENTS;
        
        // Write RGB (we ignore Alpha, as required by P6 format)
        file.put(image_data_rgba[offset_rgba + 0]); // R
        file.put(image_data_rgba[offset_rgba + 1]); // G
        file.put(image_data_rgba[offset_rgba + 2]); // B
    }

    file.close();
}

// --- Main Application Logic with Thrust ---

int main() {
    // Ensure CUDA context is initialized and ready
    CUDA_CHECK(cudaSetDevice(0));
    
    const int N = DIM * DIM;
    const int RGBA_SIZE = N * 4;

    // 1. Setup Device Vectors for GPU Calculation
    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<unsigned char> d_julia_values(N);

    // Populate indices (0, 1, 2, ... N-1) on the Host
    std::vector<int> h_indices_temp(N);
    std::iota(h_indices_temp.begin(), h_indices_temp.end(), 0);
    d_indices = h_indices_temp; // Copy indices to device

    // 2. Calculate Julia Set on GPU via Thrust::transform
    // The device iterators implicitly select the thrust::device execution policy
    thrust::transform(d_indices.begin(), d_indices.end(), 
                      d_julia_values.begin(), julia_functor(DIM));

    // 3. Transfer Results and Map Colors on Host
    
    // Copy the calculated Julia values (0 or 1) back to a host vector
    thrust::host_vector<unsigned char> h_julia_values = d_julia_values;

    // Create the final RGBA buffer (as a standard vector)
    std::vector<unsigned char> h_bitmap_rgba(RGBA_SIZE);

    // Perform color mapping safely using a standard host C++ loop
    for (int i = 0; i < N; ++i) {
        unsigned char value = h_julia_values[i];
        int offset = i * 4;

        // Color mapping logic: Red if value=1, Black if value=0, Alpha=255
        h_bitmap_rgba[offset + 0] = 255 * value; // R
        h_bitmap_rgba[offset + 1] = 0;           // G
        h_bitmap_rgba[offset + 2] = 0;           // B
        h_bitmap_rgba[offset + 3] = 255;         // A
    }

    // 4. Save Final Output to PPM File
    const char* filename = "julia_set_output.ppm";
    save_ppm_file(h_bitmap_rgba, DIM, DIM, filename); 

    // 5. Final Output and Verification
    long foreground_pixels = std::count(h_julia_values.begin(), h_julia_values.end(), 1);

    std::cout << "Julia Set Calculation Complete." << std::endl;
    std::cout << "Image size: " << DIM << "x" << DIM << " (" << N << " pixels)" << std::endl;
    std::cout << "Foreground Pixels (value=1): " << foreground_pixels << std::endl;
    std::cout << "Output saved successfully to: " << filename << std::endl;
    
    return 0;
}