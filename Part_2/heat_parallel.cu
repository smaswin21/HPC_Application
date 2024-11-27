#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Define grid dimensions and constants
#define NX 500              // Grid size in x-direction
#define NY 500              // Grid size in y-direction
#define MAX_ITER 1000       // Maximum number of iterations
#define TOLERANCE 1e-6      // Convergence tolerance

// Atomic operation for finding the maximum difference on the GPU
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CUDA kernel to update the grid and calculate the maximum difference
__global__ void updateGrid(double* u, double* u_new, int nx, int ny, double* max_diff) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Column index

    if (i < nx - 1 && j < ny - 1) {
        // Update grid point
        u_new[i * ny + j] = 0.25 * (u[(i + 1) * ny + j] + u[(i - 1) * ny + j] +
                                    u[i * ny + (j + 1)] + u[i * ny + (j - 1)]);
        double diff = fabs(u_new[i * ny + j] - u[i * ny + j]);
        atomicMaxDouble(max_diff, diff);

        // Debug: Print updated values
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("UpdateGrid: u[%d][%d] = %f, diff = %f\n", i, j, u_new[i * ny + j], diff);
        }
    }
}

// Function to write the grid data to a VTK file for visualization
void writeVTK(const char *filename, int nx, int ny, double *data) {
    FILE *file = fopen(filename, "w");
    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "2D Heat Distribution\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET STRUCTURED_POINTS\n");
    fprintf(file, "DIMENSIONS %d %d 1\n", nx, ny);
    fprintf(file, "ORIGIN 0 0 0\n");
    fprintf(file, "SPACING 1 1 1\n");
    fprintf(file, "POINT_DATA %d\n", nx * ny);
    fprintf(file, "SCALARS temperature double\n");
    fprintf(file, "LOOKUP_TABLE default\n");

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fprintf(file, "%f\n", data[i * ny + j]);
        }
    }
    fclose(file);
    printf("VTK output saved to %s\n", filename);
}

int main() {
    int nx = NX, ny = NY;  // Grid dimensions
    size_t size = nx * ny * sizeof(double);

    // Host memory allocation
    double *u = (double*)malloc(size);     // Current grid
    double *u_new = (double*)malloc(size); // Updated grid
    double *max_diff = (double*)malloc(sizeof(double)); // Maximum difference

    // Initialize the grid with boundary conditions
    printf("Initializing grid...\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            u[i * ny + j] = 0.0; // Interior points
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                u[i * ny + j] = 100.0; // Boundary points
            }
        }
    }

    // Device memory allocation
    double *d_u, *d_u_new, *d_max_diff;
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_u_new, size);
    cudaMalloc((void**)&d_max_diff, sizeof(double));

    // Copy the initial grid to the device
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);

    // Define thread and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((ny + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (nx + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Iterative computation loop
    for (int iter = 0; iter < MAX_ITER; iter++) {
        *max_diff = 0.0;
        cudaMemcpy(d_max_diff, max_diff, sizeof(double), cudaMemcpyHostToDevice);

        // Launch the kernel
        printf("Iteration %d: Launching kernel...\n", iter + 1);
        updateGrid<<<numBlocks, threadsPerBlock>>>(d_u, d_u_new, nx, ny, d_max_diff);

        // Copy the maximum difference back to the host
        cudaMemcpy(max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost);
        printf("Iteration %d: Max difference = %f\n", iter + 1, *max_diff);

        // Check for convergence
        if (*max_diff < TOLERANCE) {
            printf("Converged after %d iterations.\n", iter + 1);
            break;
        }

        // Swap the pointers for the next iteration
        double* temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;
    }

    // Stop GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total execution time: %f ms\n", milliseconds);

    // Copy the final grid back to the host
    cudaMemcpy(u, d_u, size, cudaMemcpyDeviceToHost);

    // Print a small portion of the grid
    printf("Grid sample after convergence:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%0.2f ", u[i * ny + j]);
        }
        printf("\n");
    }
    
    
    writeVTK("heat_output.vtk", NX, NY, u);

    // Free device and host memory
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_max_diff);
    free(u);
    free(u_new);
    free(max_diff);

    return 0;
}
