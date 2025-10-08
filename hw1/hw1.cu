#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1024
#endif

// meanings of status = {'0' = two slovers; '1' = 1 slover; '2' = imaginary; '3' = any; '4' = incorrect}
__global__ void solve_quadratic_batch(const float* a, const float* b, const float* c,
                                      float* out1, float* out2, int* status, int n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = blockDim.x * gridDim.x;
    for (; i <n; i += offset) {
        float A = a[i], B = b[i], C = c[i];

        // if A = 0
        if ( A == 0) {
            if ( B == 0) {
                if ( C== 0) {
                    status[i] = 3;  // any(0x + 0 = 0)
                } else {
                    status[i] = 4;  // incorrect（0x + C = 0）
                }
            } else {
                out1[i] = -C / B;
                status[i] = 1;// // (0 + Bx + C = 0 -> x = -B/C)
            }
            continue;
        }

        // if A != 0
        float D = B*B - 4.0f*A*C;
        if (D > 0) {
            float sqrtD = sqrtf(D);
            out1[i] = (-B + sqrtD) / (2.0f*A);
            out2[i] = (-B - sqrtD) / (2.0f*A);
            status[i] = 0;
        } else if (D < 0 ) {
            status[i] = 2; // imaginary
        } else {
            out1[i] = -B / (2.0f*A);
            status[i] = 1;
        }
    }
}

int main() {

    int n = 1;
    //std::cin >> n;

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> h_a[i] >> h_b[i] >> h_c[i];
    }

    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr, *d_out1=nullptr, *d_out2=nullptr;
    int *d_status=nullptr;
    cudaMalloc(&d_a, n*sizeof(float));
    cudaMalloc(&d_b, n*sizeof(float));
    cudaMalloc(&d_c, n*sizeof(float));
    cudaMalloc(&d_out1, n*sizeof(float));
    cudaMalloc(&d_out2, n*sizeof(float));
    cudaMalloc(&d_status, n*sizeof(int));

    cudaMemcpy(d_a, h_a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(NUM_BLOCKS), threads(THREADS_PER_BLOCK);
    solve_quadratic_batch<<<blocks, threads>>>(d_a, d_b, d_c, d_out1, d_out2, d_status, n);

    std::vector<float> h_out1(n), h_out2(n);
    std::vector<int> h_status(n);

    cudaMemcpy(h_out1.data(), d_out1, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2.data(), d_out2, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_status.data(), d_status, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_out1); cudaFree(d_out2); cudaFree(d_status);

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        switch (h_status[i]) {
            case 0: std::cout << h_out1[i] << " " << h_out2[i] << "\n"; break;
            case 1: std::cout << h_out1[i] << "\n"; break;
            case 2: std::cout << "imaginary\n"; break;
            case 3: std::cout << "any\n"; break;
            case 4: std::cout << "incorrect\n"; break;
        }
    }
    return 0;
}
