#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Include time.h for time measurement
#include <mpi.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#define N 10000

void matrix_multiply(int *A, int *B, int *C, int size) {
    // Your existing matrix multiplication code remains unchanged
}

int main(int argc, char **argv) {
    int rank, size;
    int *A, *B, *C;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    A = (int*)malloc(N * N * sizeof(int));
    B = (int*)malloc(N * N * sizeof(int));
    C = (int*)malloc(N * N * sizeof(int));

    if (rank == 0) {
        // Initialize matrices A and B on rank 0
    }

    size_t rows_per_process = N / size;
    int local_A = (int)malloc(rows_per_process * N * sizeof(int));
    int local_C = (int)malloc(rows_per_process * N * sizeof(int));

    MPI_Scatter(A, rows_per_process * N, MPI_INT, local_A, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure execution time
    double start_time = MPI_Wtime();

    matrix_multiply(local_A, B, local_C, rows_per_process);

    // Measure execution time
    double end_time = MPI_Wtime();

    double execution_time = end_time - start_time;
    printf("Execution time %d: %lf seconds\n", rank, execution_time);

    MPI_Gather(local_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);
    MPI_Finalize();
    return 0;
}
