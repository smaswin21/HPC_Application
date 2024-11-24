#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;  // Declare timing variables
    start_time = MPI_Wtime();    // Start timing

    // Calculate number of rows per process
    int rows_per_proc = NX / size;
    int extra_rows = NX % size;

    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int local_NX = rows_per_proc + (rank < extra_rows ? 1 : 0) + 2;  // +2 for ghost rows

    // Local arrays to hold the grid data
    double u_local[local_NX][NY];
    double u_new_local[local_NX][NY];

    // Initialize the grid including ghost rows
    for (int i = 0; i < local_NX; i++) {
        for (int j = 0; j < NY; j++) {
            u_local[i][j] = 0.0;
            if ((start_row + i - 1) == 0 || (start_row + i - 1) == NX - 1 || j == 0 || j == NY - 1) {
                u_local[i][j] = 100.0; // Boundary conditions
            }
        }
    }

    // Main iteration loop
    double max_diff;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        max_diff = 0.0;

        // Communication: Exchange boundary rows with neighboring processes
        MPI_Request reqs[4];
        int err;

        if (rank > 0) {
            err = MPI_Isend(u_local[1], NY, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(u_local[0], NY, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[1]);
        }
        if (rank < size - 1) {
            err = MPI_Isend(u_local[local_NX - 2], NY, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(u_local[local_NX - 1], NY, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);
        }

        // Wait for communication to complete
        if (rank > 0) {
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Wait(&reqs[2], MPI_STATUS_IGNORE);
            MPI_Wait(&reqs[3], MPI_STATUS_IGNORE);
        }

        // Parallel computation with OpenMP
        int i, j;
        #pragma omp parallel for reduction(max:max_diff) private(i, j)
        for (i = 1; i < local_NX - 1; i++) {
            for (j = 1; j < NY - 1; j++) {
                u_new_local[i][j] = 0.25 * (u_local[i + 1][j] + u_local[i - 1][j]
                                           + u_local[i][j + 1] + u_local[i][j - 1]);
                double diff = fabs(u_new_local[i][j] - u_local[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        // Update the local grid with the newly computed values
        #pragma omp parallel for private(i, j)
        for (i = 1; i < local_NX - 1; i++) {
            for (j = 1; j < NY - 1; j++) {
                u_local[i][j] = u_new_local[i][j];
            }
        }

        // Check for global convergence
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_diff < TOLERANCE) {
            if (rank == 0) {
                printf("Converged after %d iterations.\n", iter);
            }
            break;
        }
    }

    end_time = MPI_Wtime();  // End timing
    if (rank == 0) {
        printf("Execution time for parallel version: %f seconds\n", end_time - start_time);
    }

    // Output final results in CSV format (only rank 0 writes the data)
    if (rank == 0) {
    FILE *csv_fp = fopen("heat_output.csv", "w");
    if (csv_fp == NULL) {
        fprintf(stderr, "Error opening file for CSV output\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(csv_fp, "%f", u_local[i][j]);
            if (j < NY - 1) {
                fprintf(csv_fp, ",");
            }
        }
        fprintf(csv_fp, "\n");  // Ensure every row ends with a newline
    }

    fclose(csv_fp);
    printf("CSV file heat_output.csv written successfully.\n");
        
    }



    // Finalize MPI
    MPI_Finalize();

    return 0;
}
