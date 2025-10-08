#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1024
#endif

__global__ void kernel(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        arr[idx] *= 2.0;
        idx += offset;
    }
}

__global__ void vec_sub(double* a, double* b, double* c, size_t n) {
    size_t offset = gridDim.x * blockDim.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += offset) c[i] = a[i] - b[i];
}

int main(int argc, char** argv) {


    size_t n;
    std::cin >> n;

    std::vector<double> h_a, h_b, h_c;

    h_a.resize(n); h_b.resize(n); h_c.resize(n);
    for (size_t i = 0; i < n; ++i) std::cin >> h_a[i];
    for (size_t i = 0; i < n; ++i) std::cin >> h_b[i];

    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, sizeof(double) * n);
    cudaMemcpy(d_a, h_a.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, sizeof(double) * n);
    cudaMemcpy(d_b, h_b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMalloc(&d_c, sizeof(double) * n);

    dim3 blocks(NUM_BLOCKS);
    dim3 threads(THREADS_PER_BLOCK);

    vec_sub<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //std::cout << "Result: ";

    std::cout << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < n; ++i) {
        if (i) std::cout << ' ';
        std::cout << h_c[i];
    }
    std::cout << '\n';
    return 0;
}
