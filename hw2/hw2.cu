#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
//#ifndef NUM_BLOCKS
//#define NUM_BLOCKS 1024
//#endif

__global__ void BubbleSort(float *a,int n){

  //size_t i  = blockIdx.x * blockDim.x + threadIdx.x;
  //size_t offset = blockDim.x * gridDim.x;
    for(int i=0;i<n;i++){
        int start  = i % 2;
        for(int j=start + 2 * threadIdx.x;j<n-1;j+=2*blockDim.x){
            float x = a[j],y = a[j+1];
            if(a[j]>a[j+1]){
                a[j] = y;
                a[j+1] = x;
            }
        }
        __syncthreads();
    }
}

int main() {

    int n;
    std::cin >> n;

    float* h = (float*)std::malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; ++i) {
        std::cin >> h[i];
    }

    float *d=nullptr;
    cudaMalloc(&d, n*sizeof(float));

    cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice);
  
    dim3 threads(THREADS_PER_BLOCK);
    BubbleSort<<<1, threads>>>(d,n);

    cudaDeviceSynchronize();

    cudaMemcpy(h, d, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d);

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        std::cout << h[i] << " ";
    }
    std::cout << "\n";
    return 0;
}
