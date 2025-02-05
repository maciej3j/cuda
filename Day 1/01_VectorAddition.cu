#include <cstdio>
#include <ctime>
#define N 100000000
#define BLOCK_SIZE 256

#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


// CPU version
void AddVectorCpu(float *A, float *B, float *C, int n){
    for(int i = 0; i < n; i++){
        C[i] = A[i] + B[i];
    }
}

// GPU version
__global__ void AddVectorGpu(float *A, float *B, float *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

void InitVector(float *vec, int n){
    for (int i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // initialize vectors
    srand(time(NULL));
    InitVector(h_a, N);
    InitVector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++){
        AddVectorCpu(h_a, h_b, h_c_cpu, N);
        AddVectorGpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++){
        double start_time = get_time();
        AddVectorCpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

        printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++){
        double start_time = get_time();
        AddVectorGpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU avg time: %f\n", cpu_avg_time);
    printf("GPU avg time: %f\n", gpu_avg_time);
    printf("Speedup: x%f \n", cpu_avg_time/gpu_avg_time);
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++){
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct." : "incorrect.");


    return 0;
}