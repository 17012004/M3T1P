#include <iostream>
#include <fstream>
#include <vector>
#include <random> // Include <random> header
#include <mpi.h>
#include <CL/cl.hpp>

#define N 10 // Matrix size

using namespace std;

void generateRandomMatrix(int matrix[][N], int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

void matrix_multiply_opencl(int* A, int* B, int* C, int rows, int cols, int common_dim) {
    try {
        cl::Context context(CL_DEVICE_TYPE_GPU);

        vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0]);

        ifstream sourceFile("matrix_multiply.cl");
        string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));

        cl::Program::Sources sources(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
        cl::Program program(context, sources);

        program.build(devices);

        cl::Kernel kernel(program, "matrixMultiply");

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rows * common_dim, A);
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * common_dim * cols, B);
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * rows * cols);

        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, rows);
        kernel.setArg(4, cols);
        kernel.setArg(5, common_dim);

        cl::NDRange global(rows, cols);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * rows * cols, C);

        queue.finish();
    } catch (cl::Error err) {
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int *A, *B, *C;
    int *subA, *subC;
    int rows_per_process, rows, cols;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        rows = N;
        cols = N;

        A = new int[rows * cols];
        B = new int[cols * rows];
        C = new int[rows * cols];

        generateRandomMatrix(A, rows * cols);
        generateRandomMatrix(B, rows * cols);
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_process = rows / size;
    subA = new int[rows_per_process * cols];
    subC = new int[rows_per_process * cols];

    MPI_Scatter(A, rows_per_process * cols, MPI_INT, subA, rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

    matrix_multiply_opencl(subA, B, subC, rows_per_process, cols, cols);

    MPI_Gather(subC, rows_per_process * cols, MPI_INT, C, rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream outputFile("output_mpi_opencl.txt");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                outputFile << C[i * cols + j] << "\t";
            }
            outputFile << endl;
        }
        outputFile.close();
    }

    delete[] subA;
    delete[] subC;
    if (rank == 0) {
        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();

    return 0;
}

